#!/usr/bin/env python3
import argparse
import sys

import cv2
import numpy as np


def wczytaj_wideo(sciezka_wideo: str):
    cap = cv2.VideoCapture(sciezka_wideo)
    if not cap.isOpened():
        print(f'BŁĄD: Nie można otworzyć pliku wideo: "{sciezka_wideo}"', file=sys.stderr)
        sys.exit(1)
    return cap


def wykryj_punkty(obraz_szary: np.ndarray):
    feature_params = dict(
        maxCorners=300,
        qualityLevel=0.2,
        minDistance=7,
        blockSize=7,
    )
    punkty = cv2.goodFeaturesToTrack(obraz_szary, mask=None, **feature_params)
    if punkty is None:
        return np.empty((0, 1, 2), dtype=np.float32)
    return punkty.astype(np.float32)


def sledz_punkty(poprzedni_obraz, biezacy_obraz, poprzednie_punkty):
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    return cv2.calcOpticalFlowPyrLK(
        poprzedni_obraz, biezacy_obraz, poprzednie_punkty, None, **lk_params
    )


def rysuj_punkty(obraz, punkty):
    for punkt in punkty.reshape(-1, 2):
        x, y = punkt.ravel()
        cv2.circle(obraz, (int(x), int(y)), 4, (0, 255, 0), -1)
    return obraz


def rysuj_trajektorie(maska, stare_punkty, nowe_punkty):
    stare = stare_punkty.reshape(-1, 2)
    nowe = nowe_punkty.reshape(-1, 2)

    for nowy, stary in zip(nowe, stare):
        x_new, y_new = nowy.ravel()
        x_old, y_old = stary.ravel()
        cv2.line(
            maska,
            (int(x_new), int(y_new)),
            (int(x_old), int(y_old)),
            (0, 0, 255),
            2,
        )
    return maska


def przetwarzaj_wideo(sciezka_wideo: str):
    cap = wczytaj_wideo(sciezka_wideo)

    poprawnie, pierwsza_klatka = cap.read()
    if not poprawnie:
        print("BŁĄD: Nie można odczytać pierwszej klatki filmu", file=sys.stderr)
        return

    poprzedni_szary = cv2.cvtColor(pierwsza_klatka, cv2.COLOR_BGR2GRAY)
    poprzednie_punkty = wykryj_punkty(poprzedni_szary)
    maska_trajektorii = np.zeros_like(pierwsza_klatka)

    while True:
        poprawnie, klatka = cap.read()
        if not poprawnie:
            break

        biezacy_szary = cv2.cvtColor(klatka, cv2.COLOR_BGR2GRAY)

        nowe_punkty, status, blad = sledz_punkty(
            poprzedni_szary,
            biezacy_szary,
            poprzednie_punkty,
        )

        if nowe_punkty is None or status is None:
            poprzednie_punkty = wykryj_punkty(biezacy_szary)
            poprzedni_szary = biezacy_szary.copy()
            continue

        poprawne = status.reshape(-1) == 1
        nowe_punkty = nowe_punkty[poprawne].reshape(-1, 1, 2)
        stare_punkty = poprzednie_punkty[poprawne].reshape(-1, 1, 2)

        if nowe_punkty.shape[0] == 0:
            poprzednie_punkty = wykryj_punkty(biezacy_szary)
            poprzedni_szary = biezacy_szary.copy()
            continue

        if nowe_punkty.shape[0] < 40:
            dodatkowe = wykryj_punkty(biezacy_szary)
            if dodatkowe.shape[0] > 0:
                nowe_punkty = np.vstack((nowe_punkty, dodatkowe))
                stare_punkty = np.vstack((stare_punkty, dodatkowe))

        maska_trajektorii = rysuj_trajektorie(
            maska_trajektorii,
            stare_punkty,
            nowe_punkty,
        )

        wynik = rysuj_punkty(klatka.copy(), nowe_punkty)
        wynik = cv2.add(wynik, maska_trajektorii)

        liczba_punktow = int(nowe_punkty.shape[0])
        sredni_blad = float(np.mean(blad[poprawne])) if blad is not None and np.any(poprawne) else 0.0
        cv2.putText(
            wynik,
            f"Punkty: {liczba_punktow} | Sredni blad LK: {sredni_blad:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Optical Flow - sledzenie punktow", wynik)

        klawisz = cv2.waitKey(30) & 0xFF
        if klawisz in (ord("q"), 27):
            break

        poprzedni_szary = biezacy_szary.copy()
        poprzednie_punkty = nowe_punkty

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="LAB4 - Optical Flow")
    parser.add_argument("--video", required=True, help="Sciezka do pliku wideo")
    args = parser.parse_args()
    przetwarzaj_wideo(args.video)


if __name__ == "__main__":
    main()
