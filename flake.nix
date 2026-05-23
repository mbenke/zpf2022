{
  description = "Haskell + LaTeX development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
  };

  outputs = { self, nixpkgs }:
    let
      forAllSystems = nixpkgs.lib.genAttrs [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
    in {
      devShells = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          hspkgs = pkgs.haskell.packages.ghc98;
          tex = pkgs.texlive.combine {
            inherit (pkgs.texlive) scheme-medium
	    beamer beamerposter
	    type1cm pgf algorithms
	    a4wide;
          };
        in {
          default = pkgs.mkShell {
            packages = with hspkgs ; [
              ghc cabal-install
	      containers mtl
	      pretty
	      QuickCheck doctest
              # alex happy BNFC
	      pandoc pandoc-cli
              # tex
	      pkgs.glow
	      pkgs.pkg-config  # cabal uses this to find C libraries
	      pkgs.zlib        # C zlib for the Haskell zlib package
            ];
            # On non-NixOS the Nix glibc has no ld.so.cache. GHC's RTS links
            # against elfutils/libdw, which transitively needs zstd/xz/bzip2.
            # These can't be found at runtime without explicit LD_LIBRARY_PATH.
            #env.LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
            #  elfutils.out  # libdw.so.1
            #  zstd xz bzip2
            #]);
	    # nix-ld should have fixed it by now
          };
        });
    };
}
