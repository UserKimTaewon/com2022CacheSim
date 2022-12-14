#import argparse

#parser=argparse.ArgumentParser(prefix_chars='')
#for i in ('set','block','blocksize'):
#    parser.add_argument(i,type=int,nargs=1)


#for i in ('write-back','write-allocate','no-write-allocate','write-through'):
#    parser.add_argument(i, replace('-','_'),action='store_true')

#for i in ('fifo','lru','random'):
#    parser.add_argument(i,action='store_true')

import sys
#args=parser.parse_args()
#res2=dict(args)
set,block,blocksize=map(int,sys.argv[1:4])

baseargs=('fifo','lru','random')

write_allocate='write-allocate' in sys.argv
write_through='write-through' in sys.argv

import simulater_2 as sim
mysim=sim.make_sim(write_allocate=write_allocate,write_through=write_through,**{i:True for i in baseargs if i in sys.argv})
import sys
inp=sim.stream_to_input(sys.stdin)
loads,stores,load_hit,load_miss,store_hit,store_miss,clocks=mysim(set,block,blocksize,inp)

print(f"""
Total loads: {loads}
Total stores: {stores}
Load hits: {load_hit}
Load misses: {load_miss}
Store hits: {store_hit}
Store misses: {store_miss}
Total cycles: {clocks}
""")