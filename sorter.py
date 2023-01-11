

def momentum(asset, periodos):

    m = asset.momentum(periodos)

    return m.iloc[ -1 ]
