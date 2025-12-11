#### Kameras kalibrēšana un scēnas dziļuma noteikšana

**Mērķis:**

Apgūt dziļuma informācijas ieguvi no stereo pāriem, izmantojot disparitātes aprēķinu un kameras kalibrācijas parametrus.

**Uzdevumi:**

1\. Disparitātes karte bez kalibrācijas (Middlebury dati)

<br />

* Lejupielādēt stereo attēlu pāri no Middlebury Stereo Dataset.Piemēram: Cones, Teddy, Playroom u.c.
* Augšupielādēt attēlu pāri Colab vidē (im0.png un im1.png).
* Izveidot disparitātes karti, izmantojot cv.StereoSGBM\_create().
* Vizualizēt rezultātus.

<br />

* <br />

2\. Disparitāte un dziļuma karte ar kalibrācijas datiem

* Lejupielādēt arī kalibrācijas failu no tās pašas Middlebury mapes (calib.txt).
* Nolasīt fokusa garumu fun bāzes attālumu B.
* Aprēķināt dziļuma karti pēc formulas: Z=(f⋅B)/d
* Vizualizēt rezultātu.

3\. Disparitātes un dziļuma karte ar reāliem attēliem

<br />

* Ar savu viedtālruni uzņemt vismaz 10 foto ar šaha laukuma šablonu (no dažādiem leņķiem un attālumiem).
* Izmantot cv.findChessboardCorners() un cv.calibrateCamera() lai:
  * noteiktu kameras matricu K  un kropļojuma koeficientus dist ,
  - saglabātu rezultātu camera\_calib\_params.npz.
* Fotografēt divus attēlus (viena aina, bet kamera nobīdīta horizontāli par zināmu bāzi B , piemēram, 6 cm).
* Izveidot:
  * Disparitātes karti ar StereoSGBM
  - Dziļuma karti, izmantojot f  no kalibrācijas un B no mērījuma
* Vizualizēt un kvalitatīvi salīdzināt rezultātus ar Middlebury datiem.

**Rīki:**\
Python, OpenCV, Matplotlib

**Sagaidāmais rezultāts:**

* Programmas kods (.ipynb)
* Oriģinālie Middlebury datu kopas attēli (stereo pāris)
* Disparitātes karte (attēls) bez kalibrācijas
* Dziļuma karte ar Middlebury kalibrācijas datiem
* Savi attēli (stereo pāris) un kalibrācijas failu
* Disparitātes un dziļuma kartes no saviem attēliem
* Secinājumi (var tikt pievienoti teksta laukā, darba iesniegšanas uzdevumi, vai kā atsēvišķš pdf fails)

