import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds
import time
import cv2
import numpy as np
import os
import json
from datetime import datetime

DETECTION_MODEL_NAME = "rtdetr_traffic_sign_v8"
CLASSIFIER_MODEL_NAME = "traffic_sign_classifier_v2"

INPUT_DIR = os.environ.get("INPUT_DIR", "data/input")
BASE_OUTPUT_DIR = "/app/dataset_output"

# Tempo (em segundos) sem detecção para considerar que o objeto saiu da cena
MAX_AGE_SECONDS = 2.0

# Cache por tracker ID: acumula todas as detecções de cada objeto
# {obj_id: {"detections": [...], "last_seen": float}}
object_cache = {}
current_source = ""
current_video_index = 0
current_output_dir = ""

coco_dataset = {
    "info": {
        "description": "Traffic sign dataset - civi-embarcado",
        "version": "1.0",
        "year": datetime.now().year,
        "contributor": "civi-embarcado",
        "date_created": datetime.now().strftime("%Y/%m/%d"),
    },
    "licenses": [],
    "categories": [],
    "images": [],
    "annotations": [],
}

category_map = {}
next_annotation_id = [1]
next_image_id = [1]


def get_or_create_category(label):
    if label not in category_map:
        cat_id = len(coco_dataset["categories"]) + 1
        category_map[label] = cat_id
        coco_dataset["categories"].append({
            "id": cat_id,
            "name": label,
            "supercategory": "traffic_sign",
        })
    return category_map[label]


def finalize_object(obj_id):
    """Pega o frame do meio do histórico do objeto e salva no dataset COCO."""
    item = object_cache.pop(obj_id)
    detections = item["detections"]

    mid = len(detections) // 2
    det = detections[mid]

    frame_number = det["frame_number"]
    # Prefixo com índice do vídeo para evitar colisão de obj_id entre vídeos
    file_name = f"v{current_video_index:02d}_obj{obj_id}_frame{frame_number:08d}.jpg"
    file_path = os.path.join(current_output_dir, "images", file_name)

    cv2.imwrite(file_path, det["frame"])

    image_id = next_image_id[0]
    next_image_id[0] += 1

    coco_dataset["images"].append({
        "id": image_id,
        "file_name": file_name,
        "width": det["frame_width"],
        "height": det["frame_height"],
        "frame_number": frame_number,
        "timestamp_seconds": det["timestamp"],
        "source": current_source,
    })

    category_name = det["sub_class"] if det["sub_class"] else det["label"]
    category_id = get_or_create_category(category_name)

    coco_dataset["annotations"].append({
        "id": next_annotation_id[0],
        "image_id": image_id,
        "category_id": category_id,
        "bbox": det["bbox"],
        "area": det["bbox"][2] * det["bbox"][3],
        "iscrowd": 0,
        "score": det["confidence"],
        "attributes": {
            "detection_class": det["label"],
            "sub_class": det["sub_class"],
            "tracker_id": obj_id,
            "total_detections": len(detections),
            "middle_index": mid,
        },
    })
    next_annotation_id[0] += 1

    label_str = det["sub_class"] or det["label"]
    print(f"    [OBJETO {obj_id}] {label_str} | {len(detections)} frames | salvo: {file_name}")


def osd_sink_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    current_time = time.time()
    tempo_segundos = gst_buffer.pts / 1e9
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    # 1. Acumula detecções no cache
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        if frame_meta.obj_meta_list is None:
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
            continue

        n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        frame_copy = np.array(n_frame, copy=True, order='C')
        frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGR)
        frame_height, frame_width = frame_copy.shape[:2]

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            obj_id = obj_meta.object_id
            label = str(obj_meta.obj_label).strip() or "objeto"
            rect = obj_meta.rect_params
            top = max(int(rect.top), 0)
            left = max(int(rect.left), 0)
            bbox_width = min(int(rect.width), frame_width - left)
            bbox_height = min(int(rect.height), frame_height - top)

            sub_class = None
            if label == "traffic_sign":
                l_class = obj_meta.classifier_meta_list
                while l_class is not None:
                    try:
                        class_meta = pyds.NvDsClassifierMeta.cast(l_class.data)
                    except StopIteration:
                        break
                    l_label = class_meta.label_info_list
                    while l_label is not None:
                        try:
                            label_info = pyds.NvDsLabelInfo.cast(l_label.data)
                        except StopIteration:
                            break
                        sub_class = label_info.result_label
                        try:
                            l_label = l_label.next
                        except StopIteration:
                            break
                    try:
                        l_class = l_class.next
                    except StopIteration:
                        break

            detection = {
                "frame": frame_copy,
                "frame_number": frame_meta.frame_num,
                "timestamp": round(tempo_segundos, 3),
                "frame_width": frame_width,
                "frame_height": frame_height,
                "label": label,
                "sub_class": sub_class,
                "confidence": round(obj_meta.confidence, 4),
                "bbox": [left, top, bbox_width, bbox_height],
            }

            if obj_id not in object_cache:
                object_cache[obj_id] = {"detections": [], "last_seen": current_time}

            object_cache[obj_id]["detections"].append(detection)
            object_cache[obj_id]["last_seen"] = current_time

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    # 2. Finaliza objetos que saíram da cena
    ids_finalizados = [
        oid for oid, info in object_cache.items()
        if (current_time - info["last_seen"]) > MAX_AGE_SECONDS
    ]
    for oid in ids_finalizados:
        finalize_object(oid)

    return Gst.PadProbeReturn.OK


def flush_remaining_objects():
    """Finaliza objetos ainda no cache ao fim de cada vídeo."""
    for oid in list(object_cache.keys()):
        finalize_object(oid)


def save_coco_dataset():
    annotations_path = os.path.join(current_output_dir, "annotations.json")
    with open(annotations_path, "w", encoding="utf-8") as f:
        json.dump(coco_dataset, f, ensure_ascii=False, indent=2)
    print(f"\n[DATASET] Pasta           : {current_output_dir}")
    print(f"[DATASET] Imagens salvas  : {len(coco_dataset['images'])}")
    print(f"[DATASET] Anotações salvas: {len(coco_dataset['annotations'])}")
    print(f"[DATASET] Categorias      : {[c['name'] for c in coco_dataset['categories']]}")
    print(f"[DATASET] Arquivo COCO    : {annotations_path}")


def process_video(video_path, video_index):
    global current_source, current_video_index, object_cache, current_output_dir
    global coco_dataset, category_map

    current_source = os.path.basename(str(video_path))
    current_video_index = video_index
    object_cache = {}

    stem = os.path.splitext(current_source)[0]
    current_output_dir = os.path.join(BASE_OUTPUT_DIR, stem)
    os.makedirs(os.path.join(current_output_dir, "images"), exist_ok=True)

    # Reseta o dataset para este vídeo
    coco_dataset.clear()
    coco_dataset.update({
        "info": {
            "description": "Traffic sign dataset - civi-embarcado",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "civi-embarcado",
            "date_created": datetime.now().strftime("%Y/%m/%d"),
        },
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": [],
    })
    category_map.clear()
    next_annotation_id[0] = 1
    next_image_id[0] = 1

    from pathlib import Path as _Path
    video_uri = _Path(os.path.abspath(str(video_path))).as_uri()

    print("Construindo pipeline do DeepStream (modo dataset)...")

    pipeline  = Gst.Pipeline.new(f"pipeline-{video_index}")
    source    = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    streammux = Gst.ElementFactory.make("nvstreammux", "m")
    pgie      = Gst.ElementFactory.make("nvinfer",      "nvinfer0")
    tracker   = Gst.ElementFactory.make("nvtracker",    None)
    sgie      = Gst.ElementFactory.make("nvinfer",      "sgie")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", None)
    caps_rgba = Gst.ElementFactory.make("capsfilter",   None)
    nvosd     = Gst.ElementFactory.make("nvdsosd",      "nvdsosd0")
    nvvidconv_post = Gst.ElementFactory.make("nvvideoconvert", None)
    caps_nv12 = Gst.ElementFactory.make("capsfilter",     None)
    encoder   = Gst.ElementFactory.make("nvv4l2h264enc",  None)
    h264parse = Gst.ElementFactory.make("h264parse",      None)
    mp4mux    = Gst.ElementFactory.make("mp4mux",         None)
    filesink  = Gst.ElementFactory.make("filesink",       None)

    video_out_path = os.path.join(current_output_dir, "annotated.mp4")

    source.set_property("uri", video_uri)
    streammux.set_property("batch-size", 1)
    streammux.set_property("width", 1920)
    streammux.set_property("height", 1080)
    streammux.set_property("nvbuf-memory-type", 3)
    pgie.set_property("config-file-path", f"detection_models/{DETECTION_MODEL_NAME}/config_infer.txt")
    tracker.set_property("tracker-width", 640)
    tracker.set_property("tracker-height", 640)
    tracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
    tracker.set_property("ll-config-file", "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml")
    sgie.set_property("config-file-path", f"detection_models/{CLASSIFIER_MODEL_NAME}/config_infer.txt")
    nvvidconv.set_property("nvbuf-memory-type", 3)
    caps_rgba.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
    caps_nv12.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12"))
    encoder.set_property("bitrate", 4000000)
    filesink.set_property("location", video_out_path)
    filesink.set_property("sync", False)

    for elem in [source, streammux, pgie, tracker, sgie, nvvidconv, caps_rgba, nvosd,
                 nvvidconv_post, caps_nv12, encoder, h264parse, mp4mux, filesink]:
        pipeline.add(elem)

    for src_elem, dst_elem in [(streammux, pgie), (pgie, tracker), (tracker, sgie),
                                (sgie, nvvidconv), (nvvidconv, caps_rgba), (caps_rgba, nvosd),
                                (nvosd, nvvidconv_post), (nvvidconv_post, caps_nv12),
                                (caps_nv12, encoder), (encoder, h264parse),
                                (h264parse, mp4mux), (mp4mux, filesink)]:
        src_elem.link(dst_elem)

    def cb_newpad(src, new_pad, _):
        caps = new_pad.get_current_caps() or new_pad.query_caps(None)
        if not caps or not caps.get_structure(0).get_name().startswith("video/"):
            return  # ignora pads de áudio/legendas

        sinkpad = streammux.get_request_pad("sink_0")
        if sinkpad.is_linked():
            return

        features = caps.get_features(0)
        if features and features.contains("memory:NVMM"):
            # Decoder GPU (nvv4l2decoder) — conecta direto
            new_pad.link(sinkpad)
        else:
            # Decoder CPU — adiciona nvvideoconvert para levar para NVMM
            conv = Gst.ElementFactory.make("nvvideoconvert", None)
            conv.set_property("nvbuf-memory-type", 3)
            pipeline.add(conv)
            conv.sync_state_with_parent()
            new_pad.link(conv.get_static_pad("sink"))
            conv.get_static_pad("src").link(sinkpad)

    source.connect("pad-added", cb_newpad, 0)

    osd_pad = nvosd.get_static_pad("sink")
    if osd_pad:
        osd_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    print("Iniciando processamento para geração de dataset COCO...")
    start = time.time()
    pipeline.set_state(Gst.State.PLAYING)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_message(bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            elapsed = round(time.time() - start, 2)
            print(f"\nProcessamento concluído em {elapsed}s.")
            pipeline.set_state(Gst.State.NULL)
            flush_remaining_objects()
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Erro: {err}, {debug}")
            pipeline.set_state(Gst.State.NULL)
            loop.quit()

    bus.connect("message", on_message)

    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
        pipeline.set_state(Gst.State.NULL)
        flush_remaining_objects()
        raise


def main():
    from pathlib import Path

    videos = sorted(Path(INPUT_DIR).glob("*.mp4"))
    if not videos:
        sys.stderr.write(f"Nenhum vídeo MP4 encontrado em '{INPUT_DIR}'.\n")
        sys.exit(1)

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    Gst.init(None)

    print(f"Encontrados {len(videos)} vídeo(s) em '{INPUT_DIR}':")
    for v in videos:
        print(f"  {v.name}")

    try:
        for i, video_path in enumerate(videos):
            print(f"\n[{i + 1}/{len(videos)}] Processando: {video_path.name}")
            process_video(video_path, i)
            save_coco_dataset()
    except KeyboardInterrupt:
        print("\nInterrompido. Salvando dataset parcial...")
        save_coco_dataset()


if __name__ == "__main__":
    main()
