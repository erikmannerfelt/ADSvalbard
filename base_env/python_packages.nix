{ pkgs ? import <nixpkgs> { } }:
let
  pythonpkgs = pkgs.python310Packages.override {
    overrides = self: super: {
      scikit-gstat = super.buildPythonPackage rec {
        pname = "scikit-gstat";
        version = "1.0.8";
        src = super.fetchPypi {
          inherit pname version;
          sha256 = "5bHl6/otVWNk1r9K76KVGHLG3HgN7JcNOjqTg1jV414=";
        };
        propagatedBuildInputs = with super; [ numpy numba scipy pandas tqdm matplotlib imageio scikit-learn nose ];
        setuptoolsCheckPhase = "true";
      };

      geoutils = super.buildPythonPackage rec {
        pname = "geoutils";
        version = "0.0.9";
        src = super.fetchPypi {
          inherit pname version;
          sha256 = "9tq1Qoky6miSYOfO4iNMUXysvSW6CJ3gAkE1gCIrQkE=";
        };
        propagatedBuildInputs = with super; [
          geopandas
          tqdm
          matplotlib
          scipy
          rasterio
        ];
      };
      xdem = super.buildPythonPackage rec {
        pname = "xdem";
        version = "0.0.7";
        src = super.fetchPypi {
          inherit pname version;
          sha256 = "H+RuCeRoI3SsMITPPkhSagaJrR/+IPJPF0q/zwpPh4w=";
        };
        propagatedBuildInputs = with super; [
          opencv4
          scikit-learn
          scikitimage
          self.scikit-gstat
          self.geoutils
        ];
        setuptoolsCheckPhase = "true";
      };
      earthengine-api = super.buildPythonPackage rec {
        pname = "earthengine-api";
        version = "0.1.335";
        src = super.fetchPypi {
          inherit pname version;
          sha256 = "lix2x2Dofn5ZB/aZYJjz4f+pLw2eGCq8A4OtVPDCud0=";
        };
        propagatedBuildInputs = with super; [
          numpy
          google-cloud-storage
          google-api-python-client
          future
        ];
        setuptoolsCheckPhase = "true";
      };
      cdsapi = super.buildPythonPackage rec {
        pname = "cdsapi";
        version = "0.5.1";
        src = super.fetchPypi {
          inherit pname version;
          sha256 = "GfPpLxmWzBEV0LAoFhft6uzz7vygP704TPvFINXwR20=";
        };
        propagatedBuildInputs = with super; [
          requests
          tqdm
        ];
        setuptoolsCheckPhase = "true";
      };
      intake-xarray = super.buildPythonPackage rec {

        pname = "intake-xarray";
        version = "0.6.1";
        src = super.fetchPypi {
          inherit pname version;
          sha256 = "Y9wqi5N9j5ViVdj4ZB8I6mVLLZ6IaH7D0czGZGL6v54=";
        };
        setuptoolsCheckPhase = "true";

        propagatedBuildInputs = with super; [
            intake
            xarray
            zarr
            dask
            netcdf4
            fsspec
            msgpack
            requests
        ];

      };
      python-cmr = super.buildPythonPackage rec {
        pname = "python-cmr";
        version = "0.7.0";
        src = super.fetchPypi {
          inherit pname version;
          sha256 = "0cP6FQUxk8/epmclLaEVL4Go7Pismhg2LNz+VJ1IobU=";
        };
        setuptoolsCheckPhase = "true";
        propagatedBuildInputs = with super; [
          requests
        ];
      };
      bounded-pool-executor = super.buildPythonPackage rec {
        pname = "bounded_pool_executor";
        version = "0.0.3";
        src = super.fetchPypi {
          inherit pname version;
          sha256 = "4JIiG8OK3lVeEGSDH57YAFgPo0pLbY6d082WFUlif24=";
        };
        setuptoolsCheckPhase = "true";

      };
      pqdm = super.buildPythonPackage rec {
        pname = "pqdm";
        version = "0.2.0";
        src = super.fetchPypi {
          inherit pname version;
          sha256 = "2Z0B/kmNMntEDr/gjBTITg3J7M5hcu+aMflrsar06eM=";
        };
        setuptoolsCheckPhase = "true";
        propagatedBuildInputs = with super; [
          typing-extensions
          self.bounded-pool-executor
          tqdm
        ];
      };
      tinynetrc = super.buildPythonPackage rec {
        pname = "tinynetrc";
        version = "1.3.1";
        src = super.fetchPypi {
          inherit pname version;
          sha256 = "K5olbS5jBkO48JhfXoJszwvzcW4H5Zak9n/qs2PSVN8=";
        };
        setuptoolsCheckPhase = "true";

      };
      earthaccess = super.buildPythonPackage rec {
        pname = "earthaccess";
        version = "0.5.1";
        # The installation only works for poetry normally, so this hack overrides the setup procedure to
        # just use regular setuptools. Poetry and nix don't really mix well...
        src = pkgs.stdenv.mkDerivation {
          name = "${pname}-fixed";
          src = pkgs.fetchFromGitHub {
             owner="nsidc";
             repo=pname;
             rev="v${version}";
             sha256="bSDhZOwIGwVKSU5H8B31urSSaQoIC1JDRoM8ddrMcrk=";
          };
          new_setup = pkgs.writeText "setup.py" ''
            from setuptools import setup
            setup(
              name="${pname}",
              version="v${version}",
              packages=["${pname}", "${pname}.utils"],
            )
          '';
          buildPhase = ''
            rm poetry.lock
            rm setup.py
            rm pyproject.toml
            cp $new_setup setup.py
          '';
          installPhase = ''
            mkdir -p $out
            cp -r ./* $out
          '';
        };
        propagatedBuildInputs = with super; [
          self.python-cmr
          python-benedict
          self.pqdm
          requests
          s3fs
          fsspec
          self.tinynetrc
          multimethod

        ];

      };
      icepyx = super.buildPythonPackage rec {
        pname = "icepyx";
        version = "0.7.0";
        src = super.fetchPypi {
          inherit pname version;
          sha256 = "lVhUlURZhwjAGW5+M78CqGsxHFUW53diXQH6q3QRMrU=";
        };
        setuptoolsCheckPhase = "true";
        buildInputs = with super; [
          setuptools-scm
        ];
        propagatedBuildInputs = with super; [
          backoff
          datashader
          self.earthaccess
          geopandas
          h5netcdf
          h5py
          holoviews
          hvplot
          intake
          self.intake-xarray
          matplotlib
          requests
          xarray
        ];
      };
    };
  };


in
pythonpkgs
