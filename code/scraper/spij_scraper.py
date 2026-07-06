"""Scraper para el portal SPIJ del Ministerio de Justicia del Perú.

Ahora consulta directamente las APIs internas expuestas por el frontend.
"""
from typing import Dict, Iterable, List, Optional, Sequence
import time
import hashlib
import logging
from pathlib import Path
import re

import requests
from bs4 import BeautifulSoup

LOG = logging.getLogger(__name__)


class SpijScraper:
    """Scraper basado en las APIs internas del SPIJ."""
    # Mapeo entre los identificadores de agrupación usados por el frontend y
    # los valores efectivos que espera el endpoint de búsqueda.
    _AGRUPACION_CONFIG = {
        '1': {
            'label': 'LEGISLACIÓN DE CARACTER GENERAL',
            'agrupaciones': [
                'CONSTITUCION  POLITICA, LEYES  ORGANICAS Y CODIGOS',
                'NORMAS ADMINISTRATIVAS DE CARACTER PARTICULAR',
                'LEGISLACIÓN EMITIDA POR ENTIDADES VINCULADAS A LA ADMINISTRACIÓN DE JUSTICIA',
                'LEGISLACION SUPRANACIONAL',
                'JURISPRUDENCIA JUDICIAL, ADMINISTRATIVA, SUPRANACIONAL Y CONSTITUCIONAL',
            ],
            'tipo_norma': 'NR',
        },
        '2': {
            'label': 'LEGISLACIÓN EMITIDA POR ENTIDADES VINCULADAS A LA ADMINISTRACIÓN DE JUSTICIA',
            'agrupaciones': [
                'LEGISLACIÓN EMITIDA POR ENTIDADES VINCULADAS A LA ADMINISTRACIÓN DE JUSTICIA',
            ],
            'tipo_norma': 'NR',
        },
        '3': {
            'label': 'LEGISLACIÓN SUPRANACIONAL',
            'agrupaciones': ['LEGISLACION SUPRANACIONAL'],
            'tipo_norma': 'NR',
        },
        '4': {
            'label': 'NORMAS ADMINISTRATIVAS DE CARACTER PARTICULAR',
            'agrupaciones': ['NORMAS ADMINISTRATIVAS DE CARACTER PARTICULAR'],
            'tipo_norma': 'NR',
        },
        '5': {
            'label': 'TEXTO ÚNICO DE PROCEDIMIENTOS ADMINISTRATIVOS (TUPAS)',
            'agrupaciones': ['LEGISLACIÓN EMITIDA POR GOBIERNOS LOCALES Y REGIONALES'],
            'tipo_norma': 'NR',
        },
        '6': {
            'label': 'LEGISLACIÓN EMITIDA POR GOBIERNOS LOCALES Y REGIONALES',
            'agrupaciones': ['LEGISLACIÓN EMITIDA POR GOBIERNOS LOCALES Y REGIONALES'],
            'tipo_norma': 'NR',
        },
        '7': {
            'label': 'PERU HISTORICO',
            'agrupaciones': ['PERU HISTORICO'],
            'tipo_norma': 'NR',
            'buscar_historico': True,
        },
        '8': {
            'label': 'JURISPRUDENCIA',
            'agrupaciones': [
                'JURISPRUDENCIA JUDICIAL, ADMINISTRATIVA, SUPRANACIONAL Y CONSTITUCIONAL',
            ],
            'tipo_norma': 'JR',
        },
    }


    WEB_DETAIL_URL = 'https://spij.minjus.gob.pe/spij-ext-web/#/detallenorma/{norm_id}'
    BACK_AUTH_URL = 'https://spijwsii.minjus.gob.pe/spij-ext-back/authenticate'
    SOLR_AUTH_URL = 'https://spijwsii.minjus.gob.pe/spij-ext-solr/authenticate'
    BACK_DETAIL_URL = 'https://spijwsii.minjus.gob.pe/spij-ext-back/api/detallenorma/{norm_id}'
    SOLR_SEARCH_URL = 'https://spijwsii.minjus.gob.pe/spij-ext-solr/api/buscar'
    BACK_MAESTROS_URL = 'https://spijwsii.minjus.gob.pe/spij-ext-back/api/maestros'
    BACK_AGRUPAMIENTO_URL = 'https://spijwsii.minjus.gob.pe/spij-ext-back/api/agrupamiento'

    def __init__(self, rate_limit: float = 1.0, page_size: int = 25):
        self.rate_limit = rate_limit
        self.page_size = page_size
        self.cache_dir = Path('data/cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._robots_cache = self._fetch_robots()
        self._back_session = requests.Session()
        self._solr_session = requests.Session()
        self._maestros_cache: Optional[Dict] = None
        self._agrupaciones_cache: Optional[List[Dict]] = None
        self._authenticate()

    # --- Autenticación y catálogos -------------------------------------------------
    def _authenticate(self) -> None:
        """Obtiene y configura los tokens de autenticación requeridos."""
        LOG.debug('Autenticando contra servicios SPIJ…')
        back_resp = requests.post(
            self.BACK_AUTH_URL,
            json={'usuario': 'spijext', 'clave': 'password', 'tipo': 0},
            timeout=30,
        )
        back_resp.raise_for_status()
        back_token = back_resp.json()['value']
        self._back_session.headers.update({'Authorization': f'Bearer {back_token}'})

        solr_resp = requests.post(
            self.SOLR_AUTH_URL,
            json={'usuario': 'spijext', 'clave': 'password'},
            timeout=30,
        )
        solr_resp.raise_for_status()
        solr_token = solr_resp.json()['value']
        self._solr_session.headers.update(
            {
                'Authorization': f'Bearer {solr_token}',
                'Content-Type': 'application/json',
            }
        )

    def _get_maestros(self) -> Dict:
        if self._maestros_cache is None:
            resp = self._back_session.get(self.BACK_MAESTROS_URL, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            self._maestros_cache = data[0] if data else {}
        return self._maestros_cache

    def _get_agrupaciones(self) -> List[Dict]:
        if self._agrupaciones_cache is None:
            resp = self._back_session.get(self.BACK_AGRUPAMIENTO_URL, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            # Aplanar agrupaciones por grupo, evitando duplicados
            seen = set()
            agrupaciones: List[Dict] = []
            for block in data:
                for entry in block.get('agrupamientoNormas', []):
                    nombre = entry.get('nombre')
                    if not nombre or nombre in seen:
                        continue
                    seen.add(nombre)
                    agrupaciones.append(entry)
            self._agrupaciones_cache = agrupaciones
        return self._agrupaciones_cache

    # --- Utilidades internas -------------------------------------------------------
    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256((text or '').encode('utf-8')).hexdigest()

    @staticmethod
    def _strip_html(value: Optional[str]) -> str:
        if not value:
            return ''
        soup = BeautifulSoup(value, 'html.parser')
        return soup.get_text(separator=' ', strip=True)

    def _sleep(self) -> None:
        if self.rate_limit > 0:
            time.sleep(self.rate_limit)

    def _fetch_robots(self) -> Optional[str]:
        """Intentar descargar robots.txt y devolver su contenido (o None)."""
        try:
            url = 'https://spij.minjus.gob.pe/robots.txt'
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return r.text
        except Exception:
            LOG.debug('No se pudo obtener robots.txt')
        return None

    # --- API helpers ----------------------------------------------------------------
    def _iter_agrupacion_configs(self) -> List[Dict]:
        """Devuelve la configuración de búsqueda por agrupación que usa el frontend."""

        maestros = self._get_maestros()
        catalog = maestros.get('agrupacion', []) if maestros else []
        configs: List[Dict] = []
        seen: set[str] = set()

        for entry in catalog:
            group_id = entry.get('grupo')
            if not group_id or group_id == 'NINGUNO':
                continue
            base = self._AGRUPACION_CONFIG.get(group_id)
            if not base:
                continue
            config = {
                **base,
                'id': group_id,
                'label': entry.get('nombre') or entry.get('id') or base.get('label') or group_id,
                'agrupaciones': list(base.get('agrupaciones', [])),
            }
            configs.append(config)
            seen.add(group_id)

        # Algunos filtros (PERU HISTORICO y Jurisprudencia) no figuran siempre
        # en el catálogo principal, pero el frontend los soporta explícitamente.
        for group_id in ('7', '8'):
            if group_id in seen:
                continue
            base = self._AGRUPACION_CONFIG.get(group_id)
            if not base:
                continue
            configs.append(
                {
                    **base,
                    'id': group_id,
                    'label': base.get('label', group_id),
                    'agrupaciones': list(base.get('agrupaciones', [])),
                }
            )

        # Si nada estuvo disponible, devolver una configuración global vacía.
        if not configs:
            configs.append(
                {
                    'id': 'GLOBAL',
                    'label': 'GLOBAL',
                    'agrupaciones': [],
                    'tipo_norma': 'NR',
                    'buscar_historico': False,
                }
            )

        return configs

    def _build_payload(
        self,
        agrupaciones: Sequence[str],
        materia: Optional[Dict] = None,
        desde: int = 0,
        hasta: Optional[int] = None,
        *,
        tipo_norma: str = 'NR',
        buscar_historico: bool = False,
    ) -> Dict:
        payload = {
            'filtros': {
                'buscarHistorico': buscar_historico,
                'busquedaSugerida': False,
                'numeroDispositivoLegal': ' ',
                'dispositivoLegal': [],
                'tomo': {'id': '', 'nombre': ''},
                'materia': materia or {'id': '', 'nombre': ''},
                'agrupacion': list(agrupaciones),
                'sector': [],
                'subSector': {'id': '', 'nombre': ''},
                'orden': '1',
            },
            'facetsSeleccionadas': {'fechaPublicacionGap': {'numero': 10, 'unidad': 'YEAR'}},
            'tipoNorma': tipo_norma,
            'textoBusqueda': None,
            'textoSumilla': None,
            'desde': desde,
            'hasta': hasta if hasta is not None else desde + self.page_size,
        }
        return payload

    def _search(self, payload: Dict) -> Dict:
        resp = self._solr_session.post(self.SOLR_SEARCH_URL, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def _fetch_detail(self, norm_id: str) -> Dict:
        url = self.BACK_DETAIL_URL.format(norm_id=norm_id)
        resp = self._back_session.get(url, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def _validate_norm_status(self, norm_id: str, detail: Dict) -> tuple[bool, str]:
        """Valida si una norma sigue vigente consultando su página web.
        
        Args:
            norm_id: ID de la norma
            detail: Detalle de la norma obtenido de la API
            
        Returns:
            Tupla (es_vigente, mensaje_estado)
        """
        try:
            # Obtener la URL de la norma
            norm_url = self.WEB_DETAIL_URL.format(norm_id=norm_id)
            
            # Hacer request a la página web de la norma
            response = requests.get(norm_url, timeout=30)
            response.raise_for_status()
            
            # Parsear el HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 1. PRIMERA VERIFICACIÓN: Alertas explícitas en la página web
            alert_selectors = [
                '.alert-warning',  # Alertas amarillas
                '.alert-danger',   # Alertas rojas
                '.alert-info',     # Alertas azules
                '[class*="alert"]', # Cualquier clase que contenga 'alert'
                '.notification',
                '.warning',
                '.mensaje-estado',
                '.estado-norma'
            ]
            
            for selector in alert_selectors:
                alerts = soup.select(selector)
                for alert in alerts:
                    alert_text = alert.get_text(strip=True).lower()
                    # Patrones definitivos de invalidez
                    definitive_invalid_patterns = [
                        'esta norma ha sido derogada',
                        'norma derogada',
                        'sin efecto',
                        'no vigente',
                        'anulada',
                        'revocada',
                        'sustituida por',
                        'reemplazada por',
                        'abrogada',
                        'dejó sin efecto',
                        'ha perdido vigencia',
                        'ya no está vigente'
                    ]
                    
                    for pattern in definitive_invalid_patterns:
                        if pattern in alert_text:
                            return False, f"DEROGADA: {alert.get_text(strip=True)[:80]}"
            
            # 2. SEGUNDA VERIFICACIÓN: Patrones explícitos de derogación en encabezado
            # El SPIJ a veces marca derogaciones con "(*) DEROGADO" al inicio del texto
            # o debajo del título, sin usar una alerta amarilla.
            # Limpiamos HTML para evitar problemas con etiquetas intermedias
            raw_html = detail.get('textoCompleto', '')
            soup = BeautifulSoup(raw_html, 'html.parser')
            texto_limpio = soup.get_text(separator=' ', strip=True)
            # Usar Regex para ser flexible con espacios: (*)   DEROGADO
            # También soportar "DEJADO SIN EFECTO" o "DEJADA SIN EFECTO"
            derog_match = re.search(r'\(\*\)\s*(DEROGADO|DEJAD[OA]\s+SIN\s+EFECTO|SIN\s+EFECTO)', texto_limpio, re.IGNORECASE)
            
            if derog_match:
                # Extraer la razón
                start = derog_match.start()
                reason = texto_limpio[start:start+200]
                # Normalizar mensaje
                match_str = derog_match.group(0).upper()
                if "DEROGADO" in match_str:
                    status_type = "DEROGADA"
                else:
                    status_type = "SIN EFECTO"
                return False, f"{status_type}: {reason}"

            # Si no encontró alertas explícitas ni marcas de derogación, la norma se asume vigente
            return True, "VIGENTE"
            
        except Exception as e:
            LOG.warning(f"Error al validar estado de norma {norm_id}: {str(e)}")
            # En caso de error, asumimos que está vigente para no perder datos
            return True, f"ERROR: {str(e)[:50]}"

    def _build_item(self, doc: Dict, detail: Dict) -> Dict:
        raw_text = detail.get('textoCompleto', '')
        title_html = detail.get('titulo') or doc.get('sumilla') or ''
        title_text = self._strip_html(title_html)
        sumilla_text = self._strip_html(doc.get('sumilla')) if doc.get('sumilla') else None
        date = detail.get('fechaPublicacion') or doc.get('fechaPublicacion')
        
        # Validar estado de vigencia
        norm_id = doc.get('id')
        is_valid, status_message = self._validate_norm_status(norm_id, detail)
        
        item = {
            'type': doc.get('dispositivoLegal') or doc.get('palabra'),
            'number': doc.get('codigoNorma'),
            'title': title_text or sumilla_text,
            'date': date or None,
            'source': detail.get('sector') or doc.get('sector'),
            'status': status_message,
            'is_valid': is_valid,
            'url': self.WEB_DETAIL_URL.format(norm_id=norm_id),
            'text_hash': self._hash_text(raw_text),
            'raw_text': raw_text,
            'route': detail.get('ruta'),
            'summary_html': doc.get('sumilla'),
            'metadata': {
                'agrupacion': detail.get('ruta'),
                'solr_doc': {
                    'id': doc.get('id'),
                    'palabra': doc.get('palabra'),
                    'highlight': doc.get('highlights'),
                },
                'validation_timestamp': time.time(),
            },
        }
        return item

    # --- API-driven scraping --------------------------------------------------------
    def scrape(self, limit: int = 0, only_valid: bool = False) -> Iterable[Dict]:
        """Genera normas vía API; si limit == 0 se intenta recuperar todo.
        
        Args:
            limit: Número máximo de normas a obtener (0 = sin límite)
            only_valid: Si True, solo retorna normas que estén vigentes
        """

        seen_ids = set()
        agrupaciones = self._iter_agrupacion_configs()
        yielded = 0
        invalid_count = 0

        for agrupacion in agrupaciones:
            if limit and yielded >= limit:
                break
            filtros: Iterable[str] = agrupacion.get('agrupaciones', [])
            tipo_norma = agrupacion.get('tipo_norma', 'NR')
            buscar_historico = agrupacion.get('buscar_historico', False)
            label = agrupacion.get('label') or agrupacion.get('id')
            offset = 0
            total = None
            LOG.info('Consultando agrupación: %s', label or 'GLOBAL')
            while True:
                if limit and yielded >= limit:
                    break
                payload = self._build_payload(
                    filtros,
                    desde=offset,
                    hasta=offset + self.page_size,
                    tipo_norma=tipo_norma,
                    buscar_historico=buscar_historico,
                )
                data = self._search(payload)
                raw_total = data.get('totalEncontrados', 0)
                try:
                    total = int(raw_total)
                except (TypeError, ValueError):
                    LOG.warning('Valor inesperado para totalEncontrados=%r; usando 0', raw_total)
                    total = 0
                docs = data.get('resultados', []) or []
                if not docs:
                    break

                for doc in docs:
                    norm_id = doc.get('id')
                    if not norm_id or norm_id in seen_ids:
                        continue
                    
                    detail = self._fetch_detail(norm_id)
                    item = self._build_item(doc, detail)
                    seen_ids.add(norm_id)
                    
                    # Filtrar normas inválidas si se solicita
                    if only_valid and not item.get('is_valid', True):
                        invalid_count += 1
                        LOG.debug(f"Norma {norm_id} descartada por no estar vigente: {item.get('status')}")
                        self._sleep()
                        continue
                    
                    if not item.get('is_valid', True):
                        LOG.warning(f"Norma {norm_id} ({item.get('title', 'Sin título')[:50]}) posiblemente no vigente: {item.get('status')}")
                    
                    yielded += 1
                    yield item
                    self._sleep()
                    if limit and yielded >= limit:
                        break

                offset += self.page_size
                if total is not None and offset >= total:
                    break

        if only_valid and invalid_count > 0:
            LOG.info(f"Scraping completado. {yielded} normas válidas procesadas, {invalid_count} descartadas por no estar vigentes.")
        elif invalid_count > 0:
            LOG.info(f"Scraping completado. {yielded} normas procesadas, {invalid_count} marcadas como posiblemente no vigentes.")
        else:
            LOG.info(f"Scraping completado. {yielded} normas procesadas.")
        
        return
