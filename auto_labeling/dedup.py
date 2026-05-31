from __future__ import annotations

import numpy as np
from PIL import Image


class FrameDeduplicator:
    """Filtra frames de vídeo excessivamente semelhantes via difference hash (dhash).

    Mantém o hash do último frame *aceito*. Um frame candidato é descartado
    se a distância de Hamming em relação a esse hash for menor ou igual a
    ``max_distance``. Comparar apenas com o último aceito (não com todos os
    anteriores) é intencional: garante O(1) por frame e captura mudanças
    graduais de cena.

    Parâmetros
    ----------
    max_distance : int
        Distância de Hamming máxima para considerar dois frames duplicados.
        Intervalo válido: 0–64 (hash de 8×8 bits). Valor menor = mais
        estrito; 0 aceita apenas frames idênticos pixel a pixel.
    hash_size : int
        Dimensão do grid do dhash. O vetor de bits terá ``hash_size ** 2``
        elementos. Padrão 8 → hash de 64 bits.
    """

    def __init__(self, max_distance: int = 10, hash_size: int = 8) -> None:
        self.max_distance = max_distance
        self.hash_size = hash_size
        self._last_hash: np.ndarray | None = None

    def is_duplicate(self, image: Image.Image) -> bool:
        """Retorna ``True`` se o frame for similar demais ao último aceito.

        O primeiro frame chamado nunca é duplicata — serve de âncora inicial.
        Quando um frame é aceito (não duplicata), o hash interno é atualizado
        para ele. Quando é descartado, o hash permanece no último aceito.

        Parâmetros
        ----------
        image : PIL.Image.Image
            Frame a avaliar (qualquer modo de cor; convertido internamente).

        Retorna
        -------
        bool
            ``True`` se ``hamming(hash(image), hash(último aceito)) <= max_distance``.
        """
        current = self._dhash(image)
        if self._last_hash is None:
            self._last_hash = current
            return False
        distance = int(np.count_nonzero(current != self._last_hash))
        if distance > self.max_distance:
            self._last_hash = current
            return False
        return True

    def reset(self) -> None:
        """Descarta o hash armazenado, reiniciando o estado do deduplicador."""
        self._last_hash = None

    def _dhash(self, image: Image.Image) -> np.ndarray:
        """Calcula o difference hash da imagem.

        Redimensiona para (hash_size+1) × hash_size em escala de cinza e
        compara cada pixel com o vizinho à direita. O resultado é um vetor
        booleano de ``hash_size ** 2`` bits.
        """
        img = image.resize((self.hash_size + 1, self.hash_size), Image.LANCZOS).convert("L")
        pixels = np.array(img, dtype=np.int16)
        return (pixels[:, :-1] > pixels[:, 1:]).flatten()
