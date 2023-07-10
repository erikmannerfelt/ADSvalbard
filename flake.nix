{
  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    nixrik.url = "gitlab:erikmannerfelt/nixrik";
    nixrik.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, nixrik }:
    nixrik.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        my-python = (nixrik.packages.${system}.python_from_requirements {python_packages=pkgs.python310Packages;}) ./requirements.txt;
        packages = pkgs.lib.attrsets.recursiveUpdate (builtins.listToAttrs (map (pkg: { name = pkg.pname; value = pkg; }) (with pkgs; [
            pre-commit
            zsh
            graphviz
          ]))) {
            python=my-python;
          };

      in
      {
        inherit packages;
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
