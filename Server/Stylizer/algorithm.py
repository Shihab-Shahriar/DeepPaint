from Stylizer.AbsModels import stylizeAbstract
from Stylizer.AdaIN import stylizeAdaIN

def stylize(Ic,Is,info):
    print("INFO:",info)
    print("SHAPES:",Ic.size,Is.size)
    if 'model_name' not in info:
        return stylizeAdaIN(Ic,Is,info['slider']/10)
    else:
        return stylizeAbstract(Ic,info)

