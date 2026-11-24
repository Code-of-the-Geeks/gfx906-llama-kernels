import struct,sys
f=open(sys.argv[1],'rb')
def r(fmt): return struct.unpack('<'+fmt,f.read(struct.calcsize('<'+fmt)))[0]
def s(): n=r('Q'); return f.read(n).decode('utf-8','replace')
SC={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
def val(t):
    if t==8: return s()
    if t==9:
        et=r('I'); n=r('Q'); out=[val(et) for _ in range(n)]; return out
    return r(SC[t])
assert f.read(4)==b'GGUF'; r('I'); nt=r('Q'); nkv=r('Q')
for _ in range(nkv):
    k=s(); t=r('I'); v=val(t)
    if isinstance(v,list): v=f'[{len(v)}] '+str(v[:8])
    if k.startswith('tokenizer.ggml.') and k not in('tokenizer.ggml.model','tokenizer.ggml.pre'): continue
    if k=='tokenizer.chat_template': v=str(v)[:80]
    print(k,v)
print('n_tensors',nt)
