# Instrucciones para instalar las cosas

## Nota importante
Te recomiendo fuertemente que *jamás de los jamases* uses `pip` para instalar cosas. Si ya de plano es la única opción y agotaste todas las demás, úsalo con `pip install --user` para que no instale nada en el sistema. pip es el peor y más desagradable manejador de paquetes en existencia.

## Instalación de conda
Te recomiendo fuertemente usar Anaconda (o algún otro manejador de python environments). Si no sabes de qué hablo, usa [Anaconda](https://www.anaconda.com/). Sigue las instrucciones de ahí para instalarlo (es muy fácil).

## Instalación de los paquetes que usaremos
Una vez que tengas instalado anaconda, pon esto en tu terminal de bash:

```bash
conda create -n fastai
conda activate fastai
conda config --prepend channels conda-forge
conda config --prepend channels pytorch
conda config --prepend channels fastai/label/test
conda config --prepend channels fastai
conda install fastai pytorch pillow-simd
```

Después, espera varias horas en lo que instala todo. Mientras ve a leer [esto](https://medium.com/@karpathy/software-2-0-a64152b37c35).

## Uso

Cuando quieras trabajar en la clase, empiezas por abrir una terminal y escribir:

```bash
conda activate fastai
```

Luego, puedes por ejemplo poner al día los paquetes así:

```bash
conda update --all
```

Para abrir el jupyter:
```bash
jupyter lab
```

Ahí deberías poder trabajar agusto.