{
  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    nixrik.url = "gitlab:erikmannerfelt/nixrik";
    nixrik.inputs.nixpkgs.follows = "nixpkgs";
  };
  outputs = {self, nixpkgs, nixrik}: {
    devShells = nixrik.extra.lib.for_all_systems(pkgs_pre: (
      let
        pkgs = pkgs_pre.extend nixrik.overlays.python_extra;
        my-python = pkgs.python312PackagesExtra.from_requirements ./requirements.txt;
      in {
        default = pkgs.mkShell {
          name = "ADSvalbard";
          buildInputs = with pkgs; [
            my-python
            pdal
            pre-commit
            graphviz
            ruff
          ];
        };
      }
    ));
  };
}
