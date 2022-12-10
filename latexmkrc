# -*- mode: sh; sh-shell: bash; -*-
# Google 'latexmk' for explanation of this config file
# or see https://mg.readthedocs.io/latexmk.html
# latexmk at unix command line will compile the paper
$do_cd = 1;
$clean_ext = "bbl nav out snm dvi idv mk4 css cfg tmp xref 4tc out aux log fls fdb_latexmk synctex.gz toc svg png html 4ct ps out.ps upa upb dep";
$bibtex_use=2;
$pdf_mode = 1;
$rc_report = 1;
$pdflatex="pdflatex -interaction=nonstopmode %O %S";
$aux_out_dir_report = 1;
$silent  = 0;
#$success_cmd = "./latexmk-after-success-clean.sh";
#$failure_cmd = "./latexmk-after-failure.sh";
#warn "PATH = '$ENV{PATH}'\n";
