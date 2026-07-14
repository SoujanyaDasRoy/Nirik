"""
Model registry for the Nirikhshon backend.
Manages model metadata and availability without loading model weights.
"""
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ModelMetadata:
    """Metadata for a model."""
    name: str
    version: str = "1.0.0"
    description: str = ""
    framework: str = "TensorFlow/Keras"
    input_shape: List[int] = field(default_factory=lambda: [224, 224, 3])
    output_shape: List[int] = field(default_factory=lambda: [1])
    preprocessing: str = "standard"
    is_active: bool = False
    file_path: str = ""
    file_size: int = 0
    checksum: str = ""  # For integrity verification
    tags: List[str] = field(default_factory=list)


class ModelRegistry:
    """Registry for managing available models."""

    def __repr__(self):
        """String representation."""
        return f"<ModelRegistry models={len(self._models)} active={self._active_model}>"

    def __init__(self, model_dir: str):
        """
        Initialize the model registry.

        Args:
            model_dir: Directory where model files are stored
        """
        self.model_dir = Path(model_dir)
        self._models: Dict[str, ModelMetadata] = {}
        self._active_model: Optional[str] = None
        self._scan_models()

    def _scan_models(self) -> None:
        """Scan the model directory for model files and their metadata."""
        if not self.model_dir.exists():
            return

        # Look for common model file extensions
        model_extensions = {".keras", ".h5", ".pb", ".pt", ".pth"}
        for model_file in self.model_dir.rglob("*"):
            if model_file.is_file() and model_file.suffix.lower() in model_extensions:
                model_name = model_file.stem
                metadata = self._load_metadata(model_file)
                self._models[model_name] = metadata

    def _load_metadata(self, model_file: Path) -> ModelMetadata:
        """
        Load metadata for a model file.

        Looks for a JSON file with the same name as the model file (e.g., model.json).

        Args:
            model_file: Path to the model file

        Returns:
            ModelMetadata instance
        """
        # Default metadata
        metadata = ModelMetadata(
            name=model_file.stem,
            file_path=str(model_file),
            file_size=model_file.stat().st_size
        )

        # Try to load metadata from JSON file
        metadata_file = model_file.with_suffix(".json")
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    data = json.load(f)
                # Update metadata with values from JSON
                for key, value in data.items():
                    if hasattr(metadata, key):
                        setattr(metadata, key, value)
            except (json.JSONDecodeError, IOError):
                # If we can't read the metadata, use defaults
                pass

        return metadata

    def register_model(self, metadata: ModelMetadata) -> None:
        """
        Register a model in the registry.

        Args:
            metadata: Model metadata to register
        """
        self._models[metadata.name] = metadata

    def get_model(self, name: str) -> Optional[ModelMetadata]:
        """
        Get metadata for a model by name.

        Args:
            name: Name of the model

        Returns:
            ModelMetadata if found, None otherwise
        """
        return self._models.get(name)

    def get_all_models(self) -> Dict[str, ModelMetadata]:
        """
        Get all registered models.

        Returns:
            Dictionary mapping model names to their metadata
        """
        return self._models.copy()

    def set_active_model(self, name: str) -> bool:
        """
        Set the active model.

        Args:
            name: Name of the model to activate

        Returns:
            True if successful, False if model not found
        """
        if name in self._models:
            # Deactivate current active model
            if self._active_model and self._active_model in self._models:
                self._models[self._active_model].is_active = False
            # Activate new model
            self._models[name].is_active = True
            self._active_model = name
            return True
        return False

    def get_active_model(self) -> Optional[ModelMetadata]:
        """
        Get the currently active model.

        Returns:
            ModelMetadata of active model, or None if none is active
        """
        if self._active_model:
            return self._models.get(self._active_model)
        return None

    def is_model_available(self, name: str) -> bool:
        """
        Check if a model is available in the registry.

        Args:
            name: Name of the model to check

        Returns:
            True if model exists, False otherwise
        """
        return name in self._models

    def remove_model(self, name: str) -> bool:
        """
        Remove a model from the registry.

        Args:
            name: Name of the model to remove

        Returns:
            True if model was removed, False if not found
        """
        if name in self._models:
            del self._models[name]
            if self._active_model == name:
                self._active_model = None
            return True
        return False