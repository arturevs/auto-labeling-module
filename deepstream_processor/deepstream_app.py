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
import uuid

# DETECTION_MODEL_NAME = "rtdetr_traffic_sign_det_v1"
# CLASSIFIER_MODEL_NAME = "traffic_sign_classifier"

DETECTION_MODEL_NAME = "rtdetr_traffic_sign_v8"
CLASSIFIER_MODEL_NAME = "cnn_traffic_sign_v2"

# INPUT_VIDEO = "data/input/teste_civi_placas.mp4"
INPUT_VIDEO = "data/input/dash_cam_rj_1.mp4"

# OUTPUT_VIDEO = "data/output/exemplos_civi/teste_civi_placas_1.mp4"
OUTPUT_VIDEO = "data/output/dashcam_rj/teste_civi_placas_1.mp4"


object_cache = {}
# Tempo (em segundos) para considerar que o objeto saiu da cena
MAX_AGE_SECONDS = 2.0 

def osd_sink_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK
    current_time = time.time()
    tempo_segundos = gst_buffer.pts / 1e9
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    # 1. ATUALIZAÇÃO DO CACHE (Frames atuais)
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
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
            confianca_pct = obj_meta.confidence * 100
            label_limpo = str(obj_meta.obj_label).strip() or "objeto"
            rect = obj_meta.rect_params
            top = max(int(rect.top), 0)
            left = max(int(rect.left), 0)
            bottom = min(int(rect.top + rect.height), frame_height)
            right = min(int(rect.left + rect.width), frame_width)
            classificacao_secundaria = None
            if label_limpo == "traffic_sign":
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
                        classificacao_secundaria = label_info.result_label
                        # print(f"Classificação secundária para o objeto {obj_id}: (placa) {classificacao_secundaria}")
                        try:
                            l_label = l_label.next
                        except StopIteration:
                            break
                    try:
                        l_class = l_class.next
                    except StopIteration:
                        break
            object_cache[obj_id] = {
                "frame": frame_copy,
                "last_seen": current_time,
                "label": label_limpo,
                "metadados": {
                    "class": label_limpo,
                    "sub_class": classificacao_secundaria, # <- Agora vai para o seu JSON!
                    "confidence": round(confianca_pct, 2),
                    "time_cord": round(tempo_segundos, 3),
                    "event_id": None,
                    "bbox": {"x_min": left, "x_max": right, "y_min": bottom, "y_max": top}
                }
            }
            texto_tela = f"{label_limpo} {obj_id}"
            if classificacao_secundaria:
                texto_tela += f" [{classificacao_secundaria}]"
            obj_meta.text_params.display_text = texto_tela
            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    ids_finalizados = [
        oid for oid, info in object_cache.items() 
        if (current_time - info["last_seen"]) > MAX_AGE_SECONDS
    ]
    for oid in ids_finalizados:
        item = object_cache[oid]
        base_path = "/app/ram_data"
        label = item["label"]
        file_name = f"{label}_{oid}_{str(uuid.uuid4())[:8]}"

        filename_img = f"{base_path}/{file_name}.jpg"
        cv2.imwrite(filename_img, item["frame"])
        filename_json = f"{base_path}/{file_name}.json"
        item["metadados"]["event_id"] = file_name
        with open(filename_json, 'w', encoding='utf-8') as f:
            json.dump(item["metadados"], f, ensure_ascii=False, indent=4)
        if item["metadados"]["sub_class"] is None:
            print(f"    [OBJETO FINALIZADO] {label} ID: {oid} salvo.")
        else:
            print(f"    [OBJETO FINALIZADO] {label} ({item['metadados']['sub_class']}) ID: {oid} salvo.")
        # Remove do cache para não vazar memória
        del object_cache[oid]
    return Gst.PadProbeReturn.OK

def main():
    Gst.init(None)
    print("A construir o Pipeline do DeepStream...")
    pipeline_string = (
        f"filesrc location={INPUT_VIDEO} ! "
        "qtdemux ! h264parse ! nvv4l2decoder ! "
        "m.sink_0 nvstreammux name=m batch-size=1 width=1920 height=1080 nvbuf-memory-type=3 ! "
        f"nvinfer config-file-path=detection_models/{DETECTION_MODEL_NAME}/config_infer.txt ! "
        "nvtracker tracker-width=640 tracker-height=640 "
        "ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so "
        "ll-config-file=/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml ! "
        f"nvinfer config-file-path=detection_models/{CLASSIFIER_MODEL_NAME}/config_infer.txt name=sgie ! "
        "nvvideoconvert nvbuf-memory-type=3 ! video/x-raw(memory:NVMM), format=RGBA ! "
        "nvdsosd ! nvvideoconvert ! "
        "nvv4l2h264enc ! h264parse ! qtmux ! "
        f"filesink location={OUTPUT_VIDEO}"
    )
    pipeline = Gst.parse_launch(pipeline_string)
    if not pipeline:
        sys.stderr.write(" Erro crítico: Não foi possível criar o pipeline.\n")
        sys.exit(1)
    osd = pipeline.get_by_name("nvdsosd0") 
    if osd:
        osd_pad = osd.get_static_pad("sink")
        if osd_pad:
            osd_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    print("A iniciar a reprodução e inferência na GPU...")
    start = time.time()
    pipeline.set_state(Gst.State.PLAYING) # inicializa a gpu
    
    loop = GLib.MainLoop() # loop de processamento
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    def on_message(bus, message):
        t = message.type
        if t == Gst.MessageType.EOS: # fim do streaming
            print("\nProcessamento concluído com sucesso!")
            print(f"Tempo de Processamento: {round(time.time()-start,2)} segundos")
            pipeline.set_state(Gst.State.NULL)
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
        print("Interrompido pelo utilizador.")
        pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    main()