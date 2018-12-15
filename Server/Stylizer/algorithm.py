from Stylizer.AbsModels import stylizeAbstract
from Stylizer.AdaIN import stylizeAdaIN

def resize(Ic):
	x,y=Ic.size
	if x*y>2200*1180:
		return Ic.resize((2000,1080))
	return Ic

def stylize(Ic,Is,info):
    print("INFO:",info)
    print("SHAPES:",Ic.size,Is.size)
    
    Ic,Is = resize(Ic),resize(Is)

    if 'model_name' not in info:
        return stylizeAdaIN(Ic,Is,info['slider']/10)
    else:
        return stylizeAbstract(Ic,info)

