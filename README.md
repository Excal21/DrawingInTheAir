<h1>Írjunk a levegőben!</h1>
A projekt a <i>Mesterséges intelligencia</i> című tantárgyam beadandó feladataként készült. A program kamerakép alapján felismeri egy tárgy mozgását és a levegőben rajzolt betűt virtuálisan leüti.
<br></br>
A tárgyat a számítógép szín alapján érzékeli, majd egy konvolúciós neurális háló segítségével megállapítja, hogy milyen betűt rajzoltunk vele a levegőben. Az érzékeléshez használt minták a <i>letters</i> mappában találhatók, a <i>model</i> mappában pedig előre elkészített modellek vannak.
<p align="center">
</br>
<img src="test.gif">
</p>

<h2>A program által használt modulok</h2>

- OpenCV - https://pypi.org/project/opencv-python/
- TensorFlow - https://www.tensorflow.org
- Pynput - https://pypi.org/project/pynput/
- Pillow - https://pillow.readthedocs.io/en/stable/
- Matplotlib - https://matplotlib.org/
- Imutils - https://github.com/PyImageSearch/imutils
- NumPy - https://numpy.org/

Ezek a függőségek telepíthetők a következő paranccsal:
```console
pip install opencv-python tensorflow pynput pillow imutils numpy matplotlib
```
><span style="color: rgb(218, 54, 51); font-weight: bold;"> FONTOS </span>: A Matplotlib és a TensorFlow használata megköveteli a legfrissebb Visual C++ Redistributable meglétét. Ez letölhető a következő linkről: https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170
