{
  description = "Haskell + LaTeX development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
  };

  outputs = { self, nixpkgs }:
    let
      # Support multiple systems
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
            ];
          };
        });
    };
}