# phantom-stress

## Install 

Start docker container:

```bash
 docker run --name stress-phantom -w /home/shared -v $PWD:/home/shared -it ghcr.io/fenics/dolfinx/dolfinx:stable
```

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Run

Stretch test:
```bash
python3 stretch_test.py
```

Normal closed cylinder:
```bash
python3 normal_closed_cylinder.py
```

