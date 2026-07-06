"""Definiciones ORM para las normas.

Usamos SQLAlchemy declarative base para definir el modelo Norm.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Date, DateTime, Text, ForeignKey
from sqlalchemy.types import JSON
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Norm(Base):
    """Modelo que representa una norma extraída del SPIJ.

    Atributos principales:
    - id: PK
    - type: tipo de norma (Ley, Decreto Supremo, Resolución, etc.)
    - number: número o identificador
    - title: título
    - date: fecha de publicación (Date)
    - source: entidad emisora
    - status: estado de vigencia
    - url: enlace al texto
    - text_hash: hash del texto/metadatos para detectar cambios
    - level: nivel de importancia (1-4)
    - created_at / updated_at: timestamps
    """

    __tablename__ = 'norms'

    id = Column(Integer, primary_key=True)
    type = Column(String(128), nullable=True)
    number = Column(String(128), nullable=True)
    title = Column(String(1024), nullable=True)
    date = Column(Date, nullable=True)
    source = Column(String(256), nullable=True)
    status = Column(String(512), nullable=True)  # Aumentado para mensajes de validación más largos
    url = Column(String(2048), unique=True, nullable=False)
    text_hash = Column(String(128), nullable=True)
    level = Column(Integer, nullable=True)
    is_valid = Column(Integer, nullable=True, default=1)  # 1=válida, 0=inválida, NULL=desconocido
    validation_message = Column(Text, nullable=True)  # Mensaje detallado de validación
    validation_date = Column(DateTime, nullable=True)  # Cuándo se validó por última vez
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_hash = Column(String(128), nullable=True)
    last_updated_at = Column(DateTime, nullable=True)

    versions = relationship('NormVersion', back_populates='norm', cascade='all, delete-orphan')

    @classmethod
    def from_dict(cls, d: dict):
        """Crear instancia desde diccionario (espera campos básicos)."""
        date = d.get('date')
        if isinstance(date, str):
            try:
                date = datetime.fromisoformat(date).date()
            except Exception:
                date = None
        return cls(
            type=d.get('type'),
            number=d.get('number'),
            title=d.get('title'),
            date=date,
            source=d.get('source'),
            status=d.get('status'),
            url=d.get('url'),
            text_hash=d.get('text_hash'),
            level=d.get('level'),
            is_valid=1 if d.get('is_valid', True) else 0,
            validation_message=d.get('status') if not d.get('is_valid', True) else None,
            validation_date=datetime.utcnow(),
            last_hash=d.get('text_hash'),
            last_updated_at=datetime.utcnow(),
        )

    def to_dict(self):
        return {
            'id': self.id,
            'type': self.type,
            'number': self.number,
            'title': self.title,
            'date': self.date.isoformat() if self.date else None,
            'source': self.source,
            'status': self.status,
            'url': self.url,
            'text_hash': self.text_hash,
            'level': self.level,
            'is_valid': bool(self.is_valid) if self.is_valid is not None else None,
            'validation_message': self.validation_message,
            'validation_date': self.validation_date.isoformat() if self.validation_date else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'last_hash': self.last_hash,
            'last_updated_at': self.last_updated_at.isoformat() if self.last_updated_at else None,
        }


class NormVersion(Base):
    """Tabla de versiones de normas.
    Guarda el texto completo y snapshot de metadatos cada vez que cambia el hash.
    """
    __tablename__ = 'norm_versions'

    id = Column(Integer, primary_key=True)
    norm_id = Column(Integer, ForeignKey('norms.id'), nullable=False)
    version_date = Column(DateTime, default=datetime.utcnow)
    text_hash = Column(String(128), nullable=False)
    raw_text = Column(Text, nullable=True)
    metadata_json = Column(JSON, nullable=True)

    norm = relationship('Norm', back_populates='versions')


class NormArticle(Base):
    """Artículos individuales de una norma."""
    __tablename__ = 'norm_articles'

    id = Column(Integer, primary_key=True)
    norm_id = Column(Integer, ForeignKey('norms.id'), nullable=False)
    number = Column(String(64), nullable=True)  # "1", "12-A", etc.
    content = Column(Text, nullable=True)
    order = Column(Integer, nullable=False)  # Para mantener el orden original

    norm = relationship('Norm', back_populates='articles')


class NormCitation(Base):
    """Citas o referencias a otras normas encontradas en el texto."""
    __tablename__ = 'norm_citations'

    id = Column(Integer, primary_key=True)
    source_norm_id = Column(Integer, ForeignKey('norms.id'), nullable=False)
    target_text = Column(String(256), nullable=False)  # Texto de la cita, e.g. "Ley 29158"
    citation_type = Column(String(64), nullable=True)  # "modifies", "refers", "derogates"

    norm = relationship('Norm', back_populates='citations')


# Agregar relaciones a la clase Norm
Norm.articles = relationship('NormArticle', back_populates='norm', cascade='all, delete-orphan')
Norm.citations = relationship('NormCitation', back_populates='norm', cascade='all, delete-orphan')

