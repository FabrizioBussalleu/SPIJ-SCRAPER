#!/usr/bin/env python3
"""Punto de entrada para el scraper SPIJ.

Uso:
    python main.py init-db
    python main.py scrape
    python main.py export --format csv

Este módulo ofrece una interfaz CLI mínima para inicializar la DB,
ejecutar un scraping y exportar resultados.
"""

import argparse
import os
from pathlib import Path
import logging
import click
from datetime import datetime
from dotenv import load_dotenv
from scraper.spij_scraper import SpijScraper
from db import init_db, get_session, upsert_norm_with_version
from models import Norm, NormVersion, NormArticle, NormCitation
from parser import extract_articles, extract_citations
from categorizer import categorize_norm
from utils import setup_logging

load_dotenv()

# --- CLI con click ---
@click.group()
def cli():
    """CLI principal para SPIJ Scraper."""
    pass

@cli.command('list-versions')
def cli_list_versions():
    """
    Lista todas las versiones registradas en la tabla norm_versions, mostrando los metadatos principales en formato de tabla legible.
    Utiliza Rich si está disponible, con fallback a print simple.
    """
    setup_logging()
    logging.info("Listando versiones registradas…")
    session = get_session()
    try:
        versions = session.query(NormVersion).join(Norm).order_by(NormVersion.version_date.desc()).all()
        if not versions:
            print("No hay versiones registradas todavía.")
            return
        # Preparar datos para tabla
        rows = []
        for v in versions:
            norm = v.norm
            tipo_num = f"{norm.type or ''} {norm.number or ''}".strip()
            fecha = v.version_date.strftime('%Y-%m-%d') if v.version_date else ''
            hash_short = (v.text_hash[:10] + "...") if v.text_hash else ''
            # Estado: si es la primera versión para esa norma, "Inicial", si no "Modificado"
            estado = "Inicial"
            norm_versions = session.query(NormVersion).filter_by(norm_id=norm.id).order_by(NormVersion.version_date).all()
            if len(norm_versions) > 1 and v != norm_versions[0]:
                estado = "Modificado"
            rows.append([v.id, tipo_num, fecha, hash_short, estado])
        # Mostrar tabla con Rich si está disponible
        try:
            from rich.table import Table
            from rich.console import Console
            table = Table(title="Versiones de Normas SPIJ")
            table.add_column("ID", justify="right")
            table.add_column("Norma")
            table.add_column("Fecha versión")
            table.add_column("Hash")
            table.add_column("Estado")
            for r in rows:
                table.add_row(str(r[0]), r[1], r[2], r[3], r[4])
            console = Console()
            console.print(table)
        except ImportError:
            # Fallback a print simple
            print("ID | Norma        | Fecha versión | Hash        | Estado")
            print("-- | ------------ | ------------- | ----------- | -------")
            for r in rows:
                print(f"{r[0]:<2} | {r[1]:<12} | {r[2]:<13} | {r[3]:<11} | {r[4]}")
        logging.info(f"Total de versiones: {len(rows)}")
    except Exception as e:
        logging.error(f"Error al listar versiones: {e}")
    finally:
        session.close()
#!/usr/bin/env python3
"""Punto de entrada para el scraper SPIJ.

Uso:
    python main.py init-db
    python main.py scrape
    python main.py export --format csv

Este módulo ofrece una interfaz CLI mínima para inicializar la DB,
ejecutar un scraping y exportar resultados.
"""

import argparse
import os
from pathlib import Path
import logging
import click
from datetime import datetime

from dotenv import load_dotenv

from scraper.spij_scraper import SpijScraper
from db import init_db, get_session, upsert_norm_with_version
from models import Norm, NormVersion
from categorizer import categorize_norm
from utils import setup_logging


load_dotenv()


def cmd_init_db(args):
    init_db()
    print("Base de datos inicializada.")


    setup_logging()
    try:
        scraper = SpijScraper(rate_limit=args.rate_limit)
    except Exception as e:
        print(f"Error inicializando el scraper: {e}")
        return
    items = scraper.scrape(limit=args.limit)

    session = get_session()
    added = 0
    updated = 0
    unchanged = 0
    for item in items:
        item['level'] = categorize_norm(item)
        result = upsert_norm_with_version(item, item.get('raw_text', ''), session=session)
        if result == 'added':
            added += 1
        elif result == 'updated':
            updated += 1
        else:
            unchanged += 1
    print(f"[Resumen] Nuevas: {added} | Modificadas: {updated} | Sin cambios: {unchanged}")


    session = get_session()
    q = session.query(Norm)
    rows = [n.to_dict() for n in q.all()]
    out = Path(args.output or "exports")
    out.mkdir(parents=True, exist_ok=True)
    if args.format == 'csv':
        import pandas as pd

        df = pd.DataFrame(rows)
        path = out / "normas_export.csv"
        df.to_csv(path, index=False)
        print(f"Exportado a {path}")
    else:
        import json

        path = out / "normas_export.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"Exportado a {path}")
    



# --- CLI con click ---
@click.group()
def cli():
    """CLI principal para SPIJ Scraper."""
    pass

@cli.command('init-db')
def cli_init_db():
    """Inicializa las tablas de la base de datos."""
    setup_logging()
    try:
        init_db()
        click.echo("Base de datos inicializada.")
    except Exception as e:
        logging.error(f"Error al inicializar la base de datos: {e}")

@cli.command('migrate-db')
def cli_migrate_db():
    """Ejecuta migraciones pendientes de la base de datos."""
    setup_logging()
    try:
        from migration_add_validation import migrate_add_validation_fields
        migrate_add_validation_fields()
        click.echo("✅ Migración completada: se agregaron campos de validación.")
    except Exception as e:
        logging.error(f"Error en la migración: {e}")
        click.echo(f"❌ Error en la migración: {e}")

@cli.command('scrape')
@click.option('--limit', type=int, default=0, help='Limitar número de items (0 = sin límite)')
@click.option('--rate-limit', type=float, default=1.0, help='Segundos entre requests')
@click.option('--only-valid', is_flag=True, help='Solo procesar normas que estén vigentes')
@click.option('--validate-all', is_flag=True, help='Validar vigencia de todas las normas (más lento)')
def cli_scrape(limit, rate_limit, only_valid, validate_all):
    """Ejecuta el scraping y guarda resultados en la base de datos."""
    setup_logging()
    
    if only_valid and not validate_all:
        logging.info("Iniciando scraping SOLO con normas vigentes...")
        click.echo("🔍 Modo: Solo normas vigentes (se validará cada una)")
    elif validate_all:
        logging.info("Iniciando scraping con validación completa...")
        click.echo("🔍 Modo: Validación completa (se marcará estado de todas las normas)")
    else:
        logging.info("Iniciando scraping normal...")
        click.echo("🔍 Modo: Normal (sin validación de vigencia)")
    
    try:
        scraper = SpijScraper(rate_limit=rate_limit)
        session = get_session()
        added = updated = unchanged = total = invalid = 0
        try:
            for item in scraper.scrape(limit=limit, only_valid=only_valid):
                total += 1
                raw_text = item.pop('raw_text', '') or ''
                item['level'] = categorize_norm(item)
                
                # Mostrar progreso de validación si corresponde
                if validate_all or only_valid:
                    is_valid = item.get('is_valid', True)
                    status_msg = item.get('status', 'Vigente')
                    if not is_valid:
                        invalid += 1
                        click.echo(f"⚠️  Norma {total}: {item.get('type', '')} {item.get('number', '')} - NO VIGENTE ({status_msg[:50]}...)")
                    else:
                        click.echo(f"✅ Norma {total}: {item.get('type', '')} {item.get('number', '')} - Vigente")
                
                # 1. Guardar Norma y Versión (Lógica existente)
                outcome = upsert_norm_with_version(item, raw_text, session=session)
                
                # 2. Obtener el ID de la norma recién guardada/actualizada
                # upsert_norm_with_version hace commit, así que podemos consultar
                norm_obj = session.query(Norm).filter_by(url=item['url']).first()
                
                if norm_obj and (outcome == 'added' or outcome == 'updated'):
                    # 3. Procesar y guardar Artículos (AI Chunking)
                    articles = extract_articles(raw_text)
                    if articles:
                        # Limpiar artículos anteriores si es update
                        session.query(NormArticle).filter_by(norm_id=norm_obj.id).delete()
                        for art in articles:
                            new_art = NormArticle(
                                norm_id=norm_obj.id,
                                number=art['number'],
                                content=art['content'],
                                order=art['order']
                            )
                            session.add(new_art)
                    
                    # 4. Procesar y guardar Citas (Graph Data)
                    citations = extract_citations(raw_text)
                    if citations:
                        # Limpiar citas anteriores
                        session.query(NormCitation).filter_by(source_norm_id=norm_obj.id).delete()
                        for cit in citations:
                            new_cit = NormCitation(
                                source_norm_id=norm_obj.id,
                                target_text=cit['target_text'],
                                citation_type=cit['type']
                            )
                            session.add(new_cit)
                    
                    session.commit()

                if outcome == 'added':
                    added += 1
                elif outcome == 'updated':
                    updated += 1
                else:
                    unchanged += 1

            
            logging.info("Scraping completado. Total normas extraídas: %s", total)
            if validate_all or only_valid:
                logging.info("Normas no vigentes detectadas: %s", invalid)
            logging.info("Persistencia completada: %s nuevas, %s modificadas, %s sin cambios", added, updated, unchanged)
            
            click.echo(f"\n📊 [Resumen Final]")
            click.echo(f"   Nuevas: {added} | Modificadas: {updated} | Sin cambios: {unchanged}")
            if validate_all or only_valid:
                click.echo(f"   No vigentes: {invalid}")
            click.echo(f"   Total procesadas: {total}")
        finally:
            session.close()
    except Exception as e:
        logging.error(f"Error durante el scraping: {e}")
        click.echo(f"Error durante el scraping: {e}")

@cli.command('validate-existing')
@click.option('--limit', type=int, default=0, help='Limitar número de normas a validar (0 = todas)')
@click.option('--rate-limit', type=float, default=2.0, help='Segundos entre requests (recomendado: 2.0+)')
def cli_validate_existing(limit, rate_limit):
    """Valida el estado de vigencia de normas ya existentes en la base de datos."""
    setup_logging()
    click.echo("🔍 Validando normas existentes...")
    
    try:
        scraper = SpijScraper(rate_limit=rate_limit)
        session = get_session()
        
        # Obtener normas a validar (priorizar las que no han sido validadas)
        query = session.query(Norm).filter(
            (Norm.validation_date == None) | 
            (Norm.is_valid == None)
        ).order_by(Norm.updated_at.desc())
        
        if limit > 0:
            query = query.limit(limit)
        
        norms = query.all()
        total = len(norms)
        
        if total == 0:
            click.echo("✅ No hay normas pendientes de validar.")
            return
        
        click.echo(f"📋 Encontradas {total} normas para validar...")
        
        validated = invalid_count = valid_count = errors = 0
        
        try:
            for i, norm in enumerate(norms, 1):
                click.echo(f"🔄 [{i}/{total}] Validando: {norm.type} {norm.number}...")
                
                try:
                    # Extraer el ID de la norma desde la URL
                    norm_id = norm.url.split('/')[-1] if norm.url else None
                    if not norm_id:
                        click.echo(f"❌ No se pudo extraer ID de la URL: {norm.url}")
                        errors += 1
                        continue
                    
                    # Simular detalle para la validación
                    detail = {'textoCompleto': ''}  # Detalle mínimo
                    is_valid, status_message = scraper._validate_norm_status(norm_id, detail)
                    
                    # Actualizar en base de datos
                    norm.is_valid = 1 if is_valid else 0
                    norm.validation_message = status_message
                    norm.validation_date = datetime.utcnow()
                    session.commit()
                    
                    validated += 1
                    if is_valid:
                        valid_count += 1
                        click.echo(f"✅ Vigente")
                    else:
                        invalid_count += 1
                        click.echo(f"⚠️  NO VIGENTE: {status_message[:100]}")
                        
                except Exception as e:
                    errors += 1
                    click.echo(f"❌ Error: {str(e)[:100]}")
                    logging.error(f"Error validando norma {norm.id}: {e}")
            
            click.echo(f"\n📊 [Resumen de Validación]")
            click.echo(f"   Total validadas: {validated}")
            click.echo(f"   Vigentes: {valid_count}")
            click.echo(f"   No vigentes: {invalid_count}")
            click.echo(f"   Errores: {errors}")
            
        finally:
            session.close()
            
    except Exception as e:
        logging.error(f"Error durante la validación: {e}")
        click.echo(f"Error durante la validación: {e}")

@cli.command('revalidate-all')
@click.option('--rate-limit', type=float, default=2.0, help='Segundos entre requests (recomendado: 2.0+)')
@click.option('--force', is_flag=True, help='Forzar revalidación de todas las normas, incluso las ya validadas')
def cli_revalidate_all(rate_limit, force):
    """Re-valida todas las normas con la lógica mejorada más precisa."""
    setup_logging()
    click.echo("🔍 Re-validando todas las normas con lógica mejorada...")
    
    try:
        scraper = SpijScraper(rate_limit=rate_limit)
        session = get_session()
        
        # Obtener normas a revalidar
        if force:
            query = session.query(Norm).order_by(Norm.updated_at.desc())
            click.echo("🔄 Modo: Forzar revalidación de TODAS las normas")
        else:
            query = session.query(Norm).filter(
                Norm.validation_date == None
            ).order_by(Norm.updated_at.desc())
            click.echo("🔄 Modo: Solo normas no validadas")
        
        norms = query.all()
        total = len(norms)
        
        if total == 0:
            click.echo("✅ No hay normas para revalidar.")
            return
        
        click.echo(f"📋 Encontradas {total} normas para revalidar...")
        
        validated = invalid_count = valid_count = errors = changed = 0
        
        try:
            for i, norm in enumerate(norms, 1):
                old_status = norm.is_valid
                click.echo(f"🔄 [{i}/{total}] Revalidando: {norm.type} {norm.number}...")
                
                try:
                    # Extraer el ID de la norma desde la URL
                    norm_id = norm.url.split('/')[-1] if norm.url else None
                    if not norm_id:
                        click.echo(f"❌ No se pudo extraer ID de la URL: {norm.url}")
                        errors += 1
                        continue
                    
                    # Obtener detalle y validar con lógica mejorada
                    detail = scraper._fetch_detail(norm_id)
                    is_valid, status_message = scraper._validate_norm_status(norm_id, detail)
                    
                    # Actualizar en base de datos
                    norm.is_valid = 1 if is_valid else 0
                    norm.validation_message = status_message
                    norm.validation_date = datetime.utcnow()
                    session.commit()
                    
                    validated += 1
                    if is_valid:
                        valid_count += 1
                        if old_status == 0:  # Cambió de inválida a válida
                            changed += 1
                            click.echo(f"🔄 CAMBIÓ a VIGENTE: {status_message}")
                        else:
                            click.echo(f"✅ VIGENTE: {status_message}")
                    else:
                        invalid_count += 1
                        if old_status == 1:  # Cambió de válida a inválida  
                            changed += 1
                            click.echo(f"🔄 CAMBIÓ a NO VIGENTE: {status_message}")
                        else:
                            click.echo(f"⚠️  NO VIGENTE: {status_message}")
                        
                except Exception as e:
                    errors += 1
                    click.echo(f"❌ Error: {str(e)[:100]}")
                    logging.error(f"Error revalidando norma {norm.id}: {e}")
            
            click.echo(f"\n📊 [Resumen de Re-validación]")
            click.echo(f"   Total revalidadas: {validated}")
            click.echo(f"   ✅ Vigentes: {valid_count}")
            click.echo(f"   ⚠️  No vigentes: {invalid_count}")
            click.echo(f"   🔄 Cambios de estado: {changed}")
            click.echo(f"   ❌ Errores: {errors}")
            
        finally:
            session.close()
            
    except Exception as e:
        logging.error(f"Error durante la revalidación: {e}")
        click.echo(f"Error durante la revalidación: {e}")

@cli.command('export')
@click.option('--format', type=click.Choice(['csv', 'json']), default='csv')
@click.option('--output', type=str, default=None)
def cli_export(format, output):
    """Exporta las normas a CSV o JSON."""
    setup_logging()
    logging.info(f"Exportando datos en formato {format}...")
    try:
        session = get_session()
        q = session.query(Norm)
        rows = [n.to_dict() for n in q.all()]
        out = Path(output or "exports")
        out.mkdir(parents=True, exist_ok=True)
        if format == 'csv':
            import pandas as pd
            df = pd.DataFrame(rows)
            path = out / "normas_export.csv"
            df.to_csv(path, index=False)
            click.echo(f"Exportado a {path}")
        else:
            import json
            path = out / "normas_export.json"
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)
            click.echo(f"Exportado a {path}")
        logging.info("Exportación completada.")
    except Exception as e:
        logging.error(f"Error durante la exportación: {e}")
        click.echo(f"Error durante la exportación: {e}")


if __name__ == '__main__':
    cli()
