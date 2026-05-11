"""Script para verificar los datos de validación en la base de datos."""
import logging
from db import get_session
from models import Norm
from datetime import datetime

logging.basicConfig(level=logging.INFO)

def check_validation_data():
    session = get_session()
    try:
        # Contar normas por estado de validación
        total = session.query(Norm).count()
        valid_count = session.query(Norm).filter(Norm.is_valid == 1).count()
        invalid_count = session.query(Norm).filter(Norm.is_valid == 0).count()
        unknown_count = session.query(Norm).filter(Norm.is_valid == None).count()
        
        print(f"📊 Estado de validación en la BD:")
        print(f"   Total de normas: {total}")
        print(f"   ✅ Vigentes: {valid_count}")
        print(f"   ⚠️  No vigentes: {invalid_count}")
        print(f"   ❓ Sin validar: {unknown_count}")
        print()
        
        # Mostrar algunas normas de ejemplo
        print("📋 Muestra de normas validadas:")
        norms = session.query(Norm).filter(Norm.validation_date != None).limit(5).all()
        for norm in norms:
            status_icon = "✅" if norm.is_valid else "⚠️"
            validation_date = norm.validation_date.strftime('%Y-%m-%d %H:%M') if norm.validation_date else 'N/A'
            print(f"   {status_icon} {norm.type} {norm.number} - Validado: {validation_date}")
            if norm.validation_message and norm.validation_message != "Vigente":
                print(f"      Mensaje: {norm.validation_message[:100]}")
        
        # Mostrar normas sin validar si las hay
        unvalidated = session.query(Norm).filter(Norm.validation_date == None).limit(3).all()
        if unvalidated:
            print("\n🔄 Normas pendientes de validar:")
            for norm in unvalidated:
                print(f"   📄 {norm.type} {norm.number}")
    
    finally:
        session.close()

if __name__ == "__main__":
    check_validation_data()