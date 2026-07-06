"""Script para monitorear el progreso del scraping en tiempo real."""
import time
import logging
from db import get_session
from models import Norm
from datetime import datetime, timedelta

def monitor_scraping():
    """Monitorea el progreso del scraping mostrando estadísticas en tiempo real."""
    print("🔍 Monitor de scraping SPIJ en tiempo real")
    print("=" * 50)
    
    session = get_session()
    initial_count = session.query(Norm).count()
    print(f"📊 Normas iniciales en BD: {initial_count}")
    
    # Obtener timestamp de hace 5 minutos para normas recientes
    recent_threshold = datetime.utcnow() - timedelta(minutes=5)
    
    last_count = initial_count
    start_time = datetime.utcnow()
    
    try:
        while True:
            time.sleep(10)  # Verificar cada 10 segundos
            
            # Estadísticas generales
            total = session.query(Norm).count()
            new_count = total - last_count
            
            # Normas recientes (últimos 5 minutos)
            recent_norms = session.query(Norm).filter(
                Norm.created_at >= recent_threshold
            ).count()
            
            # Estadísticas de validación
            valid_count = session.query(Norm).filter(Norm.is_valid == 1).count()
            invalid_count = session.query(Norm).filter(Norm.is_valid == 0).count()
            
            # Calcular velocidad
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            rate = (total - initial_count) / elapsed * 60 if elapsed > 0 else 0
            
            # Mostrar progreso
            print(f"\n⏱️  {datetime.now().strftime('%H:%M:%S')}")
            print(f"📈 Total normas: {total:,} (+{new_count} desde última verificación)")
            print(f"🆕 Normas recientes (5min): {recent_norms:,}")
            print(f"✅ Vigentes: {valid_count:,} | ⚠️  No vigentes: {invalid_count:,}")
            print(f"⚡ Velocidad: {rate:.1f} normas/minuto")
            print(f"⏰ Tiempo transcurrido: {elapsed/60:.1f} minutos")
            
            if new_count == 0:
                print("🔄 Sin nuevas normas - verificando si el scraping continúa...")
            
            last_count = total
            
            # Refrescar sesión para datos actualizados
            session.close()
            session = get_session()
            
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoreo detenido por el usuario")
        final_count = session.query(Norm).count()
        print(f"📊 Total final: {final_count:,} normas")
        print(f"🆕 Normas agregadas en esta sesión: {final_count - initial_count:,}")
    finally:
        session.close()

if __name__ == "__main__":
    monitor_scraping()