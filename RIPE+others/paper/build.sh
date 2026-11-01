#!/bin/sh
# Offline build with MiKTeX: vendored packages live in texmf/, the installer is disabled so nothing is fetched.
cd "$(dirname "$0")"
OPTS="--disable-installer --include-directory=texmf -interaction=nonstopmode -file-line-error"
pdflatex $OPTS main.tex >/dev/null
bibtex main >/dev/null
pdflatex $OPTS main.tex >/dev/null
pdflatex $OPTS main.tex >/dev/null
grep -E "^\./main\.tex:[0-9]+:|^!|Output written" main.log
