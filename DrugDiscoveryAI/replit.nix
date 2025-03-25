{pkgs}: {
  deps = [
    pkgs.pango
    pkgs.libGL
    pkgs.xorg.libX11
    pkgs.xorg.libXrender
    pkgs.tk
    pkgs.tcl
    pkgs.qhull
    pkgs.pkg-config
    pkgs.gtk3
    pkgs.gobject-introspection
    pkgs.ghostscript
    pkgs.freetype
    pkgs.ffmpeg-full
    pkgs.glibcLocales
    pkgs.rapidjson
    pkgs.maeparser
    pkgs.inchi
    pkgs.eigen
    pkgs.coordgenlibs
    pkgs.comic-neue
    pkgs.catch2
    pkgs.cairo
  ];
}
