import main as ad_main
import adsvalbard.filtering
import adsvalbard.stacking

def main(region: str = "heerland"):

    ad_main.symlink_region(region=region)
    adsvalbard.filtering.generate_all_masks()
    adsvalbard.stacking.create_stack()

if __name__ == "__main__":
    main()
