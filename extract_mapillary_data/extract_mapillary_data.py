from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional convenience dependency
    load_dotenv = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "datasets" / "mapillary_natal_random"
MAPILLARY_IMAGES_URL = "https://graph.mapillary.com/images"

# Approximate city coverage: min_lon, min_lat, max_lon, max_lat.
# The script splits each city into smaller boxes because Mapillary rejects
# large bbox queries and small cells also improve spatial diversity.
@dataclass(frozen=True)
class CityCoverage:
    name: str
    slug: str
    bbox: tuple[float, float, float, float]


CITY_COVERAGES = (
    CityCoverage("Rio Branco/AC", "rio_branco", (-68.0500, -10.1000, -67.7000, -9.7500)),
    CityCoverage("Maceio/AL", "maceio", (-35.8300, -9.7500, -35.6200, -9.5000)),
    CityCoverage("Macapa/AP", "macapa", (-51.2000, -0.0500, -50.9500, 0.1500)),
    CityCoverage("Manaus/AM", "manaus", (-60.1500, -3.2000, -59.8000, -2.9000)),
    CityCoverage("Salvador/BA", "salvador", (-38.6100, -13.0200, -38.3000, -12.7800)),
    CityCoverage("Fortaleza/CE", "fortaleza", (-38.6500, -3.8900, -38.3900, -3.6900)),
    CityCoverage("Brasilia/DF", "brasilia", (-48.1500, -16.0500, -47.6500, -15.5500)),
    CityCoverage("Vitoria/ES", "vitoria", (-40.3800, -20.3800, -40.2000, -20.2200)),
    CityCoverage("Goiania/GO", "goiania", (-49.4500, -16.8500, -49.1000, -16.5500)),
    CityCoverage("Sao Luis/MA", "sao_luis", (-44.4000, -2.6500, -44.1500, -2.4500)),
    CityCoverage("Cuiaba/MT", "cuiaba", (-56.2500, -15.7500, -55.9500, -15.4500)),
    CityCoverage("Campo Grande/MS", "campo_grande", (-54.8500, -20.6500, -54.4500, -20.3000)),
    CityCoverage("Belo Horizonte/MG", "belo_horizonte", (-44.0500, -20.0500, -43.8000, -19.7500)),
    CityCoverage("Belem/PA", "belem", (-48.6000, -1.5500, -48.3500, -1.3000)),
    CityCoverage("Joao Pessoa/PB", "joao_pessoa", (-34.9500, -7.2300, -34.7800, -7.0300)),
    CityCoverage("Curitiba/PR", "curitiba", (-49.4000, -25.6500, -49.1500, -25.3000)),
    CityCoverage("Recife/PE", "recife", (-35.0200, -8.1800, -34.8400, -7.9500)),
    CityCoverage("Teresina/PI", "teresina", (-42.9500, -5.2000, -42.7000, -4.9500)),
    CityCoverage("Rio de Janeiro/RJ", "rio_de_janeiro", (-43.8000, -23.1000, -43.1000, -22.7000)),
    CityCoverage("Natal/RN", "natal", (-35.2950, -5.9250, -35.1600, -5.6900)),
    CityCoverage("Porto Alegre/RS", "porto_alegre", (-51.3500, -30.2500, -50.9500, -29.9000)),
    CityCoverage("Porto Velho/RO", "porto_velho", (-64.0000, -8.9500, -63.7500, -8.6500)),
    CityCoverage("Boa Vista/RR", "boa_vista", (-60.8500, 2.7000, -60.5500, 2.9500)),
    CityCoverage("Florianopolis/SC", "florianopolis", (-48.6500, -27.8500, -48.3500, -27.3500)),
    CityCoverage("Sao Paulo/SP", "sao_paulo", (-46.8500, -24.0500, -46.3500, -23.3500)),
    CityCoverage("Aracaju/SE", "aracaju", (-37.2000, -11.1000, -37.0000, -10.8500)),
    CityCoverage("Palmas/TO", "palmas", (-48.4500, -10.3500, -48.2000, -10.0500)),
)


@dataclass(frozen=True)
class Candidate:
    image_id: str
    lon: float
    lat: float
    compass_angle: float
    sequence_id: str | None
    captured_at: int | None
    camera_type: str | None
    thumb_url: str
    bearing_error: float | None
    source_bbox: tuple[float, float, float, float]
    source_city: str
    source_city_slug: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Baixa N imagens aleatorias e espacialmente diversas do Mapillary "
            "em capitais brasileiras, priorizando cameras frontais de veiculos."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "n",
        type=int,
        help="Quantidade de imagens a baixar.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Pasta onde as imagens e o metadata.json serao salvos.",
    )
    parser.add_argument(
        "--access-token",
        default=None,
        help="Token do Mapillary. Se omitido, usa MAPILLARY_ACCESS_TOKEN do .env/ambiente.",
    )
    parser.add_argument(
        "--min-distance-m",
        type=float,
        default=10.0,
        help="Distancia minima desejada entre imagens selecionadas.",
    )
    parser.add_argument(
        "--front-angle-tolerance",
        type=float,
        default=60.0,
        help="Erro maximo, em graus, entre a camera e o rumo da sequencia.",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=0.008,
        help="Tamanho dos micro-bboxes em graus.",
    )
    parser.add_argument(
        "--limit-per-cell",
        type=int,
        default=100,
        help="Limite de imagens pedidas ao Mapillary por micro-bbox.",
    )
    parser.add_argument(
        "--pages-per-cell",
        type=int,
        default=3,
        help="Numero maximo de paginas buscadas por micro-bbox.",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=80,
        help="Numero maximo de micro-bboxes consultados.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Pausa entre chamadas/downloads para aliviar a API.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Numero de chamadas paralelas para consultar micro-bboxes.",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=8,
        help="Numero de downloads paralelos.",
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=float,
        default=2.0,
        help="Para quando houver n vezes este valor em candidatos frontais.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semente para reproduzir a selecao aleatoria.",
    )
    parser.add_argument(
        "--thumb-size",
        choices=("1024", "2048", "original"),
        default="2048",
        help="Resolucao do thumbnail baixado.",
    )
    parser.add_argument(
        "--fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Completa com imagens perspective nao panoramicas diversas se "
            "nao houver candidatos frontais suficientes."
        ),
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Compatibilidade com a versao anterior; equivale a --fallback.",
    )
    return parser.parse_args()


def resolve_access_token(token_arg: str | None) -> str:
    if load_dotenv is not None:
        load_dotenv(REPO_ROOT / ".env")

    token = token_arg or os.getenv("MAPILLARY_ACCESS_TOKEN")
    if not token:
        raise RuntimeError(
            "Token do Mapillary nao encontrado. Passe --access-token ou defina "
            "MAPILLARY_ACCESS_TOKEN no .env/ambiente."
        )
    return token


def build_cells(
    bbox: tuple[float, float, float, float],
    cell_size: float,
) -> list[tuple[float, float, float, float]]:
    min_lon, min_lat, max_lon, max_lat = bbox
    cells: list[tuple[float, float, float, float]] = []
    lon = min_lon
    while lon < max_lon:
        lat = min_lat
        next_lon = min(lon + cell_size, max_lon)
        while lat < max_lat:
            next_lat = min(lat + cell_size, max_lat)
            cells.append((lon, lat, next_lon, next_lat))
            lat = next_lat
        lon = next_lon
    return cells


def mapillary_get(
    session: requests.Session,
    access_token: str,
    bbox: tuple[float, float, float, float],
    limit: int,
    thumb_size: str,
    pages: int,
) -> list[dict[str, Any]]:
    thumb_field = f"thumb_{thumb_size}_url" if thumb_size != "original" else "thumb_original_url"
    fields = ",".join(
        [
            "id",
            "geometry",
            "computed_geometry",
            "camera_type",
            "compass_angle",
            "computed_compass_angle",
            "captured_at",
            "sequence",
            thumb_field,
        ]
    )
    base_params = {
        "bbox": ",".join(f"{value:.6f}" for value in bbox),
        "fields": fields,
        "is_panoramic": "false",
        "limit": limit,
        "access_token": access_token,
    }
    results: list[dict[str, Any]] = []
    after: str | None = None

    for _ in range(max(1, pages)):
        params = dict(base_params)
        if after:
            params["after"] = after

        response = session.get(MAPILLARY_IMAGES_URL, params=params, timeout=30)
        if response.status_code == 400:
            print(f"  bbox recusado pela API: {params['bbox']}")
            return results
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", [])
        results.extend(data)

        after = payload.get("paging", {}).get("cursors", {}).get("after")
        if not after or len(data) < limit:
            break

    return results


def point_from_geometry(image: dict[str, Any]) -> tuple[float, float] | None:
    geometry = image.get("computed_geometry") or image.get("geometry")
    if not geometry:
        return None
    coordinates = geometry.get("coordinates") or []
    if len(coordinates) < 2:
        return None
    return float(coordinates[0]), float(coordinates[1])


def compass_from_image(image: dict[str, Any]) -> float | None:
    angle = image.get("computed_compass_angle")
    if angle is None:
        angle = image.get("compass_angle")
    if angle is None:
        return None
    return float(angle) % 360.0


def sequence_id_from_image(image: dict[str, Any]) -> str | None:
    sequence = image.get("sequence")
    if isinstance(sequence, dict):
        return str(sequence.get("id")) if sequence.get("id") else None
    if sequence:
        return str(sequence)
    return None


def bearing_degrees(start: tuple[float, float], end: tuple[float, float]) -> float:
    lon1, lat1 = map(math.radians, start)
    lon2, lat2 = map(math.radians, end)
    delta_lon = lon2 - lon1
    x = math.sin(delta_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def angle_distance(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def haversine_m(left: tuple[float, float], right: tuple[float, float]) -> float:
    lon1, lat1 = map(math.radians, left)
    lon2, lat2 = map(math.radians, right)
    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1
    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )
    return 6_371_000.0 * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def add_bearing_errors(images: list[dict[str, Any]]) -> None:
    by_sequence: dict[str, list[dict[str, Any]]] = {}
    for image in images:
        sequence_id = sequence_id_from_image(image)
        if sequence_id:
            by_sequence.setdefault(sequence_id, []).append(image)

    for sequence_images in by_sequence.values():
        sequence_images.sort(key=lambda item: item.get("captured_at") or 0)
        for index, image in enumerate(sequence_images):
            point = point_from_geometry(image)
            compass = compass_from_image(image)
            if point is None or compass is None:
                continue

            movement_bearings: list[tuple[float, float]] = []
            if index > 0:
                previous_point = point_from_geometry(sequence_images[index - 1])
                if previous_point is not None:
                    movement_bearings.append((previous_point, point))
            if index + 1 < len(sequence_images):
                next_point = point_from_geometry(sequence_images[index + 1])
                if next_point is not None:
                    movement_bearings.append((point, next_point))

            errors = [
                angle_distance(compass, bearing_degrees(start, end))
                for start, end in movement_bearings
                if haversine_m(start, end) >= 3.0
            ]
            if errors:
                image["_bearing_error"] = min(errors)


def make_candidate(
    image: dict[str, Any],
    source_bbox: tuple[float, float, float, float],
    source_city: CityCoverage,
    thumb_size: str,
) -> Candidate | None:
    point = point_from_geometry(image)
    compass = compass_from_image(image)
    thumb_field = f"thumb_{thumb_size}_url" if thumb_size != "original" else "thumb_original_url"
    thumb_url = image.get(thumb_field)
    if point is None or compass is None or not thumb_url:
        return None

    image_id = str(image.get("id") or "")
    if not image_id:
        return None

    return Candidate(
        image_id=image_id,
        lon=point[0],
        lat=point[1],
        compass_angle=compass,
        sequence_id=sequence_id_from_image(image),
        captured_at=image.get("captured_at"),
        camera_type=image.get("camera_type"),
        thumb_url=thumb_url,
        bearing_error=image.get("_bearing_error"),
        source_bbox=source_bbox,
        source_city=source_city.name,
        source_city_slug=source_city.slug,
    )


def fetch_cell(
    access_token: str,
    bbox: tuple[float, float, float, float],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    with requests.Session() as session:
        raw_images = mapillary_get(
            session=session,
            access_token=access_token,
            bbox=bbox,
            limit=args.limit_per_cell,
            thumb_size=args.thumb_size,
            pages=args.pages_per_cell,
        )
    add_bearing_errors(raw_images)
    if args.sleep > 0:
        time.sleep(args.sleep)
    return raw_images


def collect_candidates(
    access_token: str,
    n: int,
    args: argparse.Namespace,
) -> tuple[list[Candidate], list[Candidate]]:
    total_cells = 0
    cell_jobs: list[tuple[CityCoverage, tuple[float, float, float, float]]] = []
    for city in CITY_COVERAGES:
        city_cells = build_cells(city.bbox, args.cell_size)
        total_cells += len(city_cells)
        random.shuffle(city_cells)
        cell_jobs.extend((city, bbox) for bbox in city_cells[:args.max_cells])
    random.shuffle(cell_jobs)

    all_candidates: dict[str, Candidate] = {}
    frontal_candidates: dict[str, Candidate] = {}
    stop_after_frontals = max(int(n * args.candidate_multiplier), n + 10)
    stop_after_any = max(int(n * max(args.candidate_multiplier, 1.2)), n + 25)

    print(
        f"Consultando ate {len(cell_jobs)} de {total_cells} micro-bboxes "
        f"em {len(CITY_COVERAGES)} capitais brasileiras "
        f"com {args.workers} workers e ate {args.pages_per_cell} paginas por bbox..."
    )

    workers = max(1, min(args.workers, len(cell_jobs)))
    cell_iter = iter(cell_jobs)
    executor = ThreadPoolExecutor(max_workers=workers)
    pending = {}
    completed = 0
    stopped_early = False

    def submit_next() -> bool:
        try:
            city, bbox = next(cell_iter)
        except StopIteration:
            return False
        future = executor.submit(fetch_cell, access_token, bbox, args)
        pending[future] = (city, bbox)
        return True

    try:
        for _ in range(workers):
            submit_next()

        while pending:
            for future in as_completed(tuple(pending)):
                city, bbox = pending.pop(future)
                completed += 1
                try:
                    raw_images = future.result()
                except requests.RequestException as exc:
                    print(f"  [{completed}/{len(cell_jobs)}] {city.name}: erro na API: {exc}")
                else:
                    for image in raw_images:
                        candidate = make_candidate(image, bbox, city, args.thumb_size)
                        if candidate is None:
                            continue
                        if candidate.camera_type and candidate.camera_type != "perspective":
                            continue

                        all_candidates.setdefault(candidate.image_id, candidate)
                        if (
                            candidate.bearing_error is not None
                            and candidate.bearing_error <= args.front_angle_tolerance
                        ):
                            frontal_candidates.setdefault(candidate.image_id, candidate)

                    print(
                        f"  [{completed}/{len(cell_jobs)}] {city.name}: "
                        f"candidatos={len(all_candidates)} "
                        f"frontais={len(frontal_candidates)}"
                    )

                enough_frontals = len(frontal_candidates) >= stop_after_frontals
                enough_fallback = args.fallback and len(all_candidates) >= stop_after_any
                if enough_frontals or enough_fallback:
                    stopped_early = True
                    executor.shutdown(wait=False, cancel_futures=True)
                    return list(frontal_candidates.values()), list(all_candidates.values())

                submit_next()
                break
    finally:
        if not stopped_early:
            executor.shutdown(wait=True, cancel_futures=True)

    return list(frontal_candidates.values()), list(all_candidates.values())


def spatially_diverse_sample(
    candidates: list[Candidate],
    n: int,
    min_distance_m: float,
) -> list[Candidate]:
    shuffled = candidates[:]
    random.shuffle(shuffled)
    shuffled.sort(
        key=lambda item: (
            item.bearing_error if item.bearing_error is not None else 999.0,
            random.random(),
        )
    )

    selected: list[Candidate] = []
    for candidate in shuffled:
        point = (candidate.lon, candidate.lat)
        if all(
            haversine_m(point, (chosen.lon, chosen.lat)) >= min_distance_m
            for chosen in selected
        ):
            selected.append(candidate)
            if len(selected) == n:
                return selected

    if len(selected) == n:
        return selected

    relaxed_distance = min_distance_m
    remaining = [item for item in shuffled if item not in selected]
    while len(selected) < n and relaxed_distance > 0.5:
        relaxed_distance *= 0.75
        for candidate in remaining[:]:
            point = (candidate.lon, candidate.lat)
            if all(
                haversine_m(point, (chosen.lon, chosen.lat)) >= relaxed_distance
                for chosen in selected
            ):
                selected.append(candidate)
                remaining.remove(candidate)
                if len(selected) == n:
                    break

    return selected


def candidate_metadata(index: int, candidate: Candidate, file_name: str) -> dict[str, Any]:
    return {
        "file_name": file_name,
        "id": candidate.image_id,
        "lon": candidate.lon,
        "lat": candidate.lat,
        "compass_angle": candidate.compass_angle,
        "bearing_error": candidate.bearing_error,
        "sequence_id": candidate.sequence_id,
        "captured_at": candidate.captured_at,
        "camera_type": candidate.camera_type,
        "source_bbox": candidate.source_bbox,
        "source_city": candidate.source_city,
        "source_city_slug": candidate.source_city_slug,
        "selection_index": index,
    }


def download_one(
    index: int,
    total: int,
    candidate: Candidate,
    output_dir: Path,
    sleep_s: float,
) -> dict[str, Any]:
    file_name = f"mapillary_{candidate.source_city_slug}_{index:04d}_{candidate.image_id}.jpg"
    destination = output_dir / file_name
    with requests.Session() as session:
        response = session.get(candidate.thumb_url, timeout=60)
        response.raise_for_status()
    destination.write_bytes(response.content)
    if sleep_s > 0:
        time.sleep(sleep_s)
    print(
        f"[{index}/{total}] {file_name} "
        f"(erro_frontal={candidate.bearing_error})"
    )
    return candidate_metadata(index, candidate, file_name)


def download_selected(
    selected: list[Candidate],
    output_dir: Path,
    sleep_s: float,
    workers: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: list[dict[str, Any]] = []
    total = len(selected)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [
            executor.submit(download_one, index, total, candidate, output_dir, sleep_s)
            for index, candidate in enumerate(selected, start=1)
        ]
        for future in as_completed(futures):
            metadata.append(future.result())

    metadata.sort(key=lambda item: item["selection_index"])
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    started_at = time.perf_counter()
    args = parse_args()
    if args.n <= 0:
        raise ValueError("n precisa ser maior que zero.")
    if args.seed is not None:
        random.seed(args.seed)
    if args.allow_fallback:
        args.fallback = True

    access_token = resolve_access_token(args.access_token)
    frontal_candidates, all_candidates = collect_candidates(access_token, args.n, args)
    pool = frontal_candidates[:]

    if len(pool) < args.n and args.fallback:
        known_ids = {candidate.image_id for candidate in pool}
        pool.extend(candidate for candidate in all_candidates if candidate.image_id not in known_ids)
        print(
            "Aviso: completando com imagens perspective nao panoramicas porque "
            "nao houve frontais suficientes."
        )

    selected = spatially_diverse_sample(pool, args.n, args.min_distance_m)
    if len(selected) < args.n:
        city_cell_counts = [
            len(build_cells(city.bbox, args.cell_size))
            for city in CITY_COVERAGES
        ]
        total_cells = sum(city_cell_counts)
        consulted_cells = sum(min(args.max_cells, count) for count in city_cell_counts)
        cell_hint = ""
        if consulted_cells < total_cells:
            max_cells_hint = max(city_cell_counts)
            cell_hint = (
                f" Com --cell-size {args.cell_size}, as {len(CITY_COVERAGES)} "
                f"capitais tem {total_cells} micro-bboxes; esta execucao "
                f"consultou no maximo {consulted_cells}. Tente --max-cells "
                f"{max_cells_hint} para varrer todas as areas."
            )
        raise RuntimeError(
            f"Somente {len(selected)} imagens atenderam aos filtros. "
            f"Foram coletados {len(all_candidates)} candidatos perspective, "
            f"incluindo {len(frontal_candidates)} candidatos frontais. "
            "Tente aumentar --max-cells, --pages-per-cell ou reduzir --min-distance-m."
            f"{cell_hint}"
        )

    print(f"Baixando {len(selected)} imagens em {args.output_dir}...")
    download_selected(selected, args.output_dir, args.sleep, args.download_workers)
    elapsed = time.perf_counter() - started_at
    print(f"Concluido em {elapsed:.1f}s.")
    print(f"Metadata salvo em {args.output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
