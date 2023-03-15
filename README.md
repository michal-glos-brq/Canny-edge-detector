### Jak spustit aplikaci?
Pro automatickou instalaci python virtuálního prostředí spusťte skript `init.sh`, který vám do pracovního adresáře nainstaluje virtuální prostředí pro Python3 (doporučujeme verzi 3.9, jinak program nemusí být kompatibilní). Po instalaci se virtuální prostředí automaticky spustí.

Pro instalaci prostředí s následným spuštěním spusťte v terminálů příkaz:
`bash ./init.sh`
Pokud chcete virtuální prostředí pouze spustit spusťte v terminálů příkaz:
`. zpoVenv/bin/activate`

Spuštění programů včetně možností spuštění je blíže specifikováno v přiložené dokumentaci.


### How to run the app?
For automatic installation of python virtualenv, use `init.sh` script, which installs the virtual environment for Python3 (3.9 version recommended, otherwise the compatibility could not be granted) in working directory. After the installation, the virtualenv will automatically run.

To install and run the virtualenv, run in terminal:
`bash ./init.sh`
To run the virtualenv, run it terminal:
`source zpoVenv/bin/activate`

### Launch options?
Gauss noise specification
 - `--b-sigma B_SIGMA` Gaussion filter kernel deviation
 - `--b-sigma-c B_SIGMA_COEF` Coefficient for dynamically computed sigma to be multiplied with
Edge detection (Sobel)
 - `--rgb` Apply filter throught all 3 channels (it's converted into greyscale otherwise)
 - `--g-ksize G_KSIZE` Sobel filter kernel size
Double tresholding
 - `--htr HTR` High treshold value
 - `--ltr LTR` Low treshold value
Other options
 - `input` path to picture to be processed (required)
 - `--grid`
 - `--step`
 - `--show`
 - `--save`
 - `--max-resolution WIDTH HEIGHT` max resolution of image SHOWN
 - `--output FILENAME, -o FILENAME` filename for output (suffix yet to be added)

