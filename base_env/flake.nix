{
  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        python_packages = import ./python_packages.nix { inherit pkgs; };
        python_from_requirements = requirements_path: (
          let
            raw_requirements = (pkgs.lib.splitString "\n" (pkgs.lib.readFile requirements_path));
            requirements = builtins.map (s: builtins.replaceStrings [ " " ] [ "" ] s) (builtins.filter (s: (builtins.stringLength s) > 1) raw_requirements);
          in
          python_packages.python.withPackages (_: map (req: (builtins.getAttr req python_packages)) requirements)
        );

        #python = python_from_requirements ./requirements.txt;

        packages = pkgs.lib.attrsets.recursiveUpdate
          (builtins.listToAttrs (map (pkg: { name = pkg.pname; value = pkg; }) (with pkgs; [
            pre-commit
            zsh
            graphviz
          ])))
          {
            inherit python_from_requirements python_packages;
          };


      in
      {
        inherit packages;
        #defaultPackage = packages.python;

        devShell = pkgs.mkShell {
          name = "ADSvalbard";
          buildInputs = pkgs.lib.attrValues packages;
          shellHook = ''

            zsh
          '';
        };
      }

    );
}
