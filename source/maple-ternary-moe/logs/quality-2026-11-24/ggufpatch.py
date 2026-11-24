# copy dst-GGUF with selected KV pairs taken (raw) from src-GGUF; tensor infos/data copied unchanged
import struct,sys,shutil
SCS={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
def parse(p):
    f=open(p,'rb'); hdr=f.read(24); nt,nkv=struct.unpack('<QQ',hdr[8:])
    def rd(n): return f.read(n)
    def skipval(t,buf):
        if t==8: b=rd(8); buf+=b; buf+=rd(struct.unpack('<Q',b)[0]); return
        if t==9:
            b=rd(12); buf+=b; et,n=struct.unpack('<IQ',b)
            for _ in range(n): skipval(et,buf)
            return
        buf+=rd(SCS[t])
    kvs=[]
    for _ in range(nkv):
        b=rd(8); kl=struct.unpack('<Q',b)[0]; k=rd(kl); tb=rd(4); t=struct.unpack('<I',tb)[0]
        buf=bytearray(b+k+tb); skipval(t,buf); kvs.append((k.decode(),bytes(buf)))
    ti_start=f.tell()
    for _ in range(nt):
        kl=struct.unpack('<Q',rd(8))[0]; rd(kl); nd=struct.unpack('<I',rd(4))[0]; rd(8*nd+4+8)
    ti_end=f.tell(); f.seek(ti_start); tib=f.read(ti_end-ti_start)
    align=32
    for k,b in kvs:
        if k=='general.alignment': align=struct.unpack('<I',b[-4:])[0]
    data_off=(ti_end+align-1)//align*align
    return hdr,nt,kvs,tib,align,data_off
src,dst,out=sys.argv[1:4]; keys=sys.argv[4].split(',')
_,_,skv,_,_,_=parse(src); sd=dict(skv)
hdr,nt,kvs,tib,align,doff=parse(dst)
kv=[(k,sd[k] if k in keys else b) for k,b in kvs]
have={k for k,_ in kv}; kv+= [(k,sd[k]) for k in keys if k not in have]
with open(out,'wb') as o:
    o.write(hdr[:8]+struct.pack('<QQ',nt,len(kv)))
    for _,b in kv: o.write(b)
    o.write(tib); pos=o.tell(); o.write(b'\0'*((pos+align-1)//align*align-pos))
    with open(dst,'rb') as i:
        i.seek(doff); shutil.copyfileobj(i,o,64<<20)
print('wrote',out,'kv',len(kvs),'->',len(kv),'align',align)
