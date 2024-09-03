# Volante virtual

## Descripción

Este proyecto tiene como objetivo crear un volante virtual que permite controlar la dirección de un carro en un juego. Utiliza la detección de poses mediante la biblioteca `Mediapipe` para capturar los puntos de referencia de las manos y calcular el ángulo de inclinación entre ellos. A partir de este ángulo, se emulan pulsaciones de teclas para controlar la dirección del vehículo.

> .[!NOTE].
> Este proyecto aún está en construcción

## Características

- Detección de pose y landmarks de las manos en tiempo real.
- Cálculo del ángulo de inclinación entre los puntos de las manos.
- Implementación de un volante virtual que permite girar a la izquierda y a la derecha.
- Soporte para la emulación de pulsaciones de teclas en función del ángulo.

## Requisitos

Para ejecutar este proyecto, necesitarás tener instaladas las siguientes bibliotecas de Python:

- OpenCV
- Mediapipe
- NumPy
- PyAutoGUI (para la emulación de teclas, si se implementa más adelante)

Puedes instalar estas bibliotecas utilizando `pip`:

```bash
pip install -r requirements.txt
```

## Instalación
1. Clona este repositorio en tu máquina local:
```bash
git clone https://github.com/brianrscode/volante-virtual.git
```

2. Navega a la carpeta del proyecto:
```bash
cd volante-virtual
```

3. Crea un entorno virtual:

    ```bashsh
    python -m venv venv
    ```

3.1. Activa el entorno virtual:

    - En Windows:

        ```bash
        venv\Scripts\activate
        ```

    - En macOS y Linux:

        ```bash
        source venv/bin/activate
        ```

4. Instala las dependencias:

    ```sh
    pip install -r requirements.txt
    ```

## Uso
1. Ejecuta el script principal:

```bash
python main.py
```

2. Colócate frente a la cámara cuidando que tus manos sean visibles, esto dibujará una línea de mano a mano y te mostrará el ángulo de inclinación de esta línea, lo que más adelante servirá para emular ciertas teclas del teclado dependiendo al ángulo de inclinación de la línea

## Contribuciones

¡Las contribuciones son bienvenidas! Siéntete libre de abrir un issue o enviar un pull request.