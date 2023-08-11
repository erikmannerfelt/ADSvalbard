import adsvalbard.utilities
import adsvalbard.arcticdem
import shapely.geometry

def process(region: str = "nordenskiold"):

    bounds = adsvalbard.utilities.get_bounds(region=region)
    strips = adsvalbard.arcticdem.get_strips(cache_label=region)


    poi = shapely.geometry.box(*list(bounds))

    print(poi)

    ...
