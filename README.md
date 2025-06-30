# CS336 Primavera 2025 Tarea 1: Fundamentos

## Descripción del Repositorio

Este repositorio contiene la implementación de los fundamentos de un modelo de lenguaje basado en transformadores para la asignación CS336 de Stanford. Incluye implementaciones de tokenización BPE, bloques de transformadores, atención multi-cabeza, y componentes esenciales para el entrenamiento de modelos de lenguaje. El proyecto está diseñado para proporcionar una base sólida para comprender y trabajar con arquitecturas de transformadores modernas.

Para una descripción completa de la tarea, consulta el documento de la asignación en
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

Si encuentras algún problema con el documento de la asignación o el código, no dudes en
crear un issue en GitHub o abrir un pull request con una corrección.

## Configuración

### Entorno
Gestionamos nuestros entornos con `uv` para garantizar reproducibilidad, portabilidad y facilidad de uso.
Instala `uv` [aquí](https://github.com/astral-sh/uv) (recomendado), o ejecuta `pip install uv`/`brew install uv`.
Te recomendamos leer un poco sobre la gestión de proyectos en `uv` [aquí](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (¡no te arrepentirás!).

Ahora puedes ejecutar cualquier código en el repositorio usando
```sh
uv run <ruta_del_archivo_python>
```
y el entorno se resolverá y activará automáticamente cuando sea necesario.

### Ejecutar pruebas unitarias

```sh
uv run pytest
```

Inicialmente, todas las pruebas deberían fallar con `NotImplementedError`s.
Para conectar tu implementación a las pruebas, completa las
funciones en [./tests/adapters.py](./tests/adapters.py).

### Descargar datos
Descarga los datos de TinyStories y una muestra de OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

