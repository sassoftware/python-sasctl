#!/usr/bin/env python
# encoding: utf-8
#
# Copyright © 2022, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import sys

def findPackageBytecodes(inst):
    from types import CodeType, FunctionType, ModuleType
    from importlib import import_module
    from functools import reduce
    
    if inst.opname == "IMPORT_NAME":
        path = inst.argval.split(".")
        path[0] = [import_module(path[0])]
        result = reduce(lambda x, a: x + [getattr(x[-1], a)], path)
        return ("modules", result)
    if inst.opname == "LOAD_GLOBAL":
        if inst.argval in globals() and type(globals()[inst.argval]) in [CodeType, FunctionType]:
            return ("code", globals()[inst.argval])
        if inst.argval in globals() and type(globals()[inst.argval]) == ModuleType:
            return ("modules", [globals()[inst.argval]])
        else:
            return None
    if "LOAD_" in inst.opname and type(inst.argval) in [CodeType, FunctionType]:
        return ("code", inst.argval)
    return None

def getFunctionPackages(func):
    import inspect
    from dis import Bytecode
    workList = [func]
    seen = set()
    mods = set()
    logging.debug("workList type = %s, len = %d" % (type(workList), len(workList)))
    logging.debug(workList)
    for fn in workList:
        if (hasattr(fn,"__name__")):
            if fn.__name__ == "getFunctionPackages":
                logging.debug("Preemptively skip %s " % fn.__name__)
                workList.remove(fn)
            elif fn.__name__ == "findPackageBytecodes":
                logging.debug("Preemptively skip %s " % fn.__name__)
                workList.remove(fn)
            else:
                logging.debug("Letting through name %s" % fn.__name__)
        else:
            print("Letting through %s" % fn)
    for fn in workList:
        codeworkList = [fn]
        logging.debug("In workList :", end='')
        logging.debug(fn)
        try:
            closureVars = inspect.getclosurevars(fn).globals
            for var, ref in closureVars.items():
                if (var != 'getFunctionPackages'):
                    if inspect.ismodule(ref):
                        mods.add(ref.__name__)
                    elif inspect.isfunction(ref) and id(ref) not in seen:
                        seen.add(id(ref))
                        mods.add(ref.__module__)
                        if (hasattr(ref, '__name__')):
                            if fn.__name__ == "getFunctionPackages":
                                logging.debug("A- Not adding to workList %s " % fn.__name__)
                            elif fn.__name__ == "findPackageBytecodes":
                                logging.debug("A- Not adding to workList %s " % fn.__name__)
                            else:
                                logging.debug("A- Adding to workList: %s" % ref)
                                workList.append(ref)
                        else:   
                            logging.debug("A- Adding to workList: %s" % ref)
                            workList.append(ref)
                    elif hasattr(ref, "__module__"):
                        logging.debug("1 fn=%s, k=%s, v=%s" % (fn, var, ref.__module__))
                        mods.add(ref.__module__)
            for block in codeworkList:
                for (bType, bRef) in [findPackageBytecodes(inst) for inst in Bytecode(block) if findPackageBytecodes(inst)]:
                    if bType == "modules":
                        newmods = [mod.__name__ for mod in bRef if hasattr(mod, "__name__")]
                        for mo in newmods:
                            if mo == "getFunctionPackages":
                                logging.debug("Also skip %s " % mo)
                                newmods.remove(mo)
                            elif mo == "findPackageBytecodes":
                                logging.debug("Also skip %s " % mo)
                                newmods.remove(mo)
                            else:
                                logging.debug("2 fn=%s, k=%s, bRef=%s" % (fn, bType, mo))
                        mods.update(set(newmods))
                    elif bType == "code" and id(bRef) not in seen:
                        seen.add(id(bRef))
                        if hasattr(bRef, "__module__"):
                            if bRef.__name__ != "getFunctionPackages" or bRef.__name__ != "findPackageBytecodes" :
                                logging.debug("3 fn=%s, bType=%s, bRef=%s" % (fn, bType, bRef))
                                mods.add(bRef.__module__)
                    if(inspect.isfunction(bRef)):
                        if (hasattr(bRef, '__name__')):
                            if bRef.__name__.find("getFunctionPackages") != -1:
                                logging.debug("B- Not adding to workList %s " % bRef.__name__)
                            elif bRef.__name__.find("findPackageBytecodes") != -1:
                                logging.debug("B- Not adding to workList %s " % bRef.__name__)
                            else:
                                logging.debug("B- Adding to workList 1: %s" % bRef.__name__)
                                workList.append(bRef)
                        else:   
                            logging.debug("B- Adding to workList 2: %s" % bRef)
                            logging.debug(dir(bRef))
                            workList.append(bRef)
                    elif(inspect.iscode(bRef)):
                        codeworkList.append(bRef)
        except TypeError:
            continue
    result = list(mods)
    result.sort()
    return result

def findDependencies():
    dependencies = ""
    mySymbols = []
    logging.debug(globals().keys())
    try:
        mySymbols = list(globals().keys())
        mySymbols.remove("getFunctionPackages")
        mySymbols.remove("findPackageBytecodes")
        for symbol in mySymbols:
            if symbol.startswith("__"):
                logging.debug("removing %s" % symbol)
                mySymbols.remove(symbol)
            if symbol.startswith("'"):
                logging.debug("removing %s" % symbol)
                mySymbols.remove(symbol)
        allList = []
        for symbol in mySymbols:
            logging.debu("Symbol=%s" % symbol)
            subList = getFunctionPackages(globals()[symbol])
            allList = allList + subList
            logging.debug('%s : %s' % (symbol, allList))
        uniqueList = set(allList)
        dependencies = ",".join(uniqueList)
    except:
        logging.debug(sys.exc_info()[0] + " occurred.")
        dependencies = ",".join(sys.exc_info())
    symbols = ",".join(mySymbols)
    return symbols, dependencies