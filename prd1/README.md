#### Attēlu pirmsapstrāde un segmentācija

**Mērķis:**\
Apgūt pamata attēlu pirmsapstrādes un segmentācijas operācijas, sagatavojot datus turpmākai analīzei (piem., kalibrācijai, dziļuma noteikšanai).

**Uzdevumi:**

1. Ielādēt 5 attēlus.
2. Veikt histogrammas izlīdzināšanu (pelēktoņiem + CLAHE krāsu attēlam).
3. Piemērot trokšņu filtrus (Gaussian, Median) un salīdzināt rezultātus.
4. Veikt segmentāciju:
   * globālā sliekšņošana,
   * Otsu metode,
   * Ūdensšķirtnes algoritms,
   * Canny kontūru noteikšana.
5. Saglabāt rezultātus un sagatavot īsu komentāru (kurš variants bija vispiemērotākais konkrētajam attēlam un kāpēc).

**Rīki:** Python, OpenCV (`cv2`), Matplotlib.

**Sagaidāmais rezultāts:** Notīrīti, ar uzlabotu kontrastu un segmentēti attēls ar dažādām maskām/variantiem. Īss komentārs par metožu salīdzinājumu.