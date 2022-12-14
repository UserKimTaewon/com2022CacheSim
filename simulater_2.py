import numpy as np
import numba

CacheLine = np.dtype([('dirty', np.bool8), ('valid', np.bool8), ('block', np.uint32)])
TInput = np.dtype([('load', np.bool8), ('addr', np.uint32)])
LruData = np.dtype([('next', np.uint32), ('prev', np.uint32)])


def _handle_line(line):
    load, addr, *_ = line.split(' ', 2)
    load = load == 'l'
    if addr.startswith('0x'):
        addr = int(addr[2:], base=16)
    else:
        addr = int(addr)
    return load, addr


def file_to_input(fname):
    with open(fname) as f:
        return stream_to_input(f)
        # return np.rec.array(np.fromiter(map(_handle_line,f),dtype=TInput))


def stream_to_input(stream):
    # return np.rec.array(np.fromiter(map(_handle_line,stream),dtype=TInput))
    return np.fromiter(map(_handle_line, stream), dtype=TInput)


@numba.njit(inline='always')
def init_fifo(set):
    return np.zeros((set,), dtype=np.uint32)


@numba.njit(inline='always')
def evac_fifo(state, line, setidx):
    state[setidx] += 1
    state[setidx] %= line
    return state[setidx]


@numba.njit(inline='always')
def evac_random(line, cache, setidx):
    for idx, cacheat in enumerate(cache[setidx]):
        if not cache.valid:
            return idx
    return np.random.randint(0, line)


@numba.njit(inline='always')
def evac_lru(cache, setidx):
    clone = cache[setidx].copy()
    cache[setidx][1:] = clone[:-1]
    cache[setidx][0] = clone[-1]
    return 0


@numba.njit(inline='always')
def get_lru(cache, setidx, block):
    lineno = _find_lineno(cache, setidx, block)
    if lineno < 0:
        return -1
    else:
        if lineno != 0:
            clone = cache[setidx].copy()
            cache[setidx][1:lineno + 1] = clone[:lineno]
            cache[setidx][0] = clone[lineno]
        return 0


@numba.njit(inline='always')  # (parallel=True)
def _find_lineno(cache, setidx, block):
    for lineno, cell in enumerate(cache[setidx]):
        if cell['block'] == block and cell['valid']:
            return lineno
    return -1


# numbainput=numba.from_dtype(TInput)
from dataclasses import dataclass

@dataclass
class RetData:
    loads: int
    stores: int
    load_hit: int
    load_miss: int
    store_hit: int
    store_miss: int
    clocks: int






def make_sim(write_allocate: bool = False, write_through: bool = True, fifo: bool = False, lru: bool = False,
             random: bool = False,clock_on_memory:bool=True,debug:bool = False):
    # if not(random or lru or fifo):
    #    raise Exception()
    if sum([random, lru, fifo]) != 1:
        raise Exception(f"only one of random,lru,fifo must be True, but {sum([random, lru, fifo])} was True.")
    if not write_through and not write_allocate:
        raise Exception("not sane to write-back without write-allocate")

    # def _sim(set:int,line:int,bytesize:int,input:...):
    #    cache=np.rec.array(np.zeros((set,line),dtype=CacheLine))
    #    return RetData(*sim(set,line,bytesize,input,cache))

    @numba.njit(inline='always')
    def get_lineno(cache, setidx, block):
        if lru:
            return get_lru(cache, setidx, block)
        else:
            return _find_lineno(cache, setidx, block)

    if not debug:
        @numba.njit(inline='always')
        def log(args):
            ...
    else:
        def log(args):
            print(*args)

    # @numba.njit(nogil=True)
    def sim(set: int, line: int, bytesize: int, input: ...):
        memtime = (bytesize // 4) * 100
        setdiv = bytesize
        setmask = setdiv * (set - 1)
        # tagdiv=setdiv*set
        # tagmask=tagdiv*(line-1)
        tagmask = 0xffff_ffff ^ (setmask | (bytesize - 1))
        log(('setdiv=', setdiv, 'setmask=', setmask, 'tagmask=', tagmask))
        # cache=create_cache(set,line)
        cache = np.zeros((set, line), dtype=CacheLine)
        clocks = 0
        loads = 0
        stores = 0
        load_hit = 0
        load_miss = 0
        store_hit = 0
        store_miss = 0
        if fifo:
            state = init_fifo()

        def handle_evac(setidx):
            lineno = -1
            if lru:
                # lineno=evac_lru(lrudata,latest,setidx)
                lineno = evac_lru(cache, setidx)
            elif fifo:
                lineno = evac_fifo(state, line, setidx)
            elif random:
                lineno = evac_random(line, cache, setidx)
            else:
                raise Exception()
            return lineno

        for inp in input:
            setidx = (inp.addr & setmask) // setdiv
            block = (inp.addr & tagmask) #원래는 block의 앞부분만 저장하는 것이 맞지만, 편의를 위해 그냥 mask만 씌워서 저장했다.
            log(('setidx=', setidx, 'block=', block))
            if inp.load:  # load인 경우에..
                loads += 1
                lineno = get_lineno(cache, setidx, block)
                # if lineno>=0: #cache hit인 경우에...
                #    if lru:
                #        #touch_lru(lrudata,latest,setidx,lineno)
                if lineno >= 0:
                    load_hit += 1
                    if not clock_on_memory:
                        clocks+=1
                else:  # cache miss인 경우에...
                    load_miss += 1
                    clocks += memtime  # 읽어오느라 시간이 걸림.
                    lineno = handle_evac(setidx)
                    if not write_through:
                        if cache[setidx][lineno]['dirty']:
                            clocks += memtime  # write-back하느라 시간이 걸림.
                    cache[setidx][lineno]['valid'] = True
                    cache[setidx][lineno]['block'] = block  # 캐시에 올림.
                    if not write_through:
                        cache[setidx][lineno]['dirty']=False
            else:  # store인 경우에...
                stores += 1
                lineno = get_lineno(cache, setidx, block)
                if lineno >= 0:
                    store_hit += 1
                    if write_through:
                        clocks += memtime
                        ...
                    else:
                        cache[setidx][lineno]['dirty'] = True
                        if not clock_on_memory:
                            clocks+=1
                    # if lru:
                    #    touch_lru(lrudata,latest,setidx,lineno)
                else:
                    store_miss += 1
                    clocks+=memtime
                    if write_allocate:
                        lineno = handle_evac(setidx)
                        if not write_through:
                            if cache[setidx][lineno]['dirty']:
                                clocks += memtime  # write-back하느라 시간이 걸림.
                        cache[setidx][lineno]['valid'] = True
                        cache[setidx][lineno]['block'] = block  # 캐시에 올림.
                        if not write_through:  # write-back이라면 dirty만 참으로 하고 끝나고,
                            cache[setidx][lineno]['dirty'] = True
                        else:
                            cache[setidx][lineno]['dirty']=False
                            clocks += memtime  # 아니면 메모리에 쓰느라 시간이 더 걸림
                    else:
                        clocks += memtime
            if clock_on_memory:
                clocks += 1
        # if not write_through:
        #    for lines in cache:
        #        for line in lines:
        #            if line['dirty']:
        #                clocks+=memtime
        return (loads, stores, load_hit, load_miss, store_hit, store_miss, clocks)

    if not debug:
        sim = numba.njit(nogil=True)(sim)

    return sim


if __name__ == '__main__':
    debug=False
    mysim = make_sim(write_allocate=True, write_through=False, lru=True,debug=debug)
    mydata = file_to_input('gcc.trace')
    print(mysim(256, 4, 16, mydata))
    #mysim.inspect_types()
