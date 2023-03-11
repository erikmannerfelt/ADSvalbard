{
  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    base_env.url = "path:./base_env";
    base_env.inputs.nixpkgs.follows = "nixpkgs";
    base_env.inputs.flake-utils.follows = "flake-utils";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, base_env, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        packages = base_env.packages.${system};

        updated_packages = pkgs.lib.filterAttrs (k: v: (k != "python_from_requirements") && (k != "python_packages")) (
          pkgs.lib.attrsets.recursiveUpdate packages {
            python=(packages.python_from_requirements ./requirements.txt);
          }
        );
      in
      {
        packages = updated_packages;
        devShell = pkgs.mkShell {
            name = "ADSvalbard";
            buildInputs = pkgs.lib.attrValues updated_packages;
            shellHook = ''

              zsh
            '';
          };
      }
  
    );
}
