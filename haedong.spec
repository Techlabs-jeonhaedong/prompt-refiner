# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'numpy', 'pandas', 'matplotlib', 'scipy', 'sklearn',
        'IPython', 'jupyter', 'notebook', 'nbformat', 'nbconvert',
        'PIL', 'Pillow', 'cv2', 'torch', 'tensorflow',
        'sphinx', 'docutils', 'babel', 'pytest', 'black',
        'tkinter', '_tkinter', 'Tkinter',
        'zmq', 'tornado', 'gevent', 'lxml',
        'cryptography', 'bcrypt', 'paramiko',
        'psutil', 'cloudpickle', 'jedi', 'parso',
        'matplotlib.backends', 'matplotlib.pyplot',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='haedong',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    console=True,
)
