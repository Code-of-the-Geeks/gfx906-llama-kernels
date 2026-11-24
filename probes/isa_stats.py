import re,sys,collections
S=sys.argv[1]
K='_ZL13mul_mat_vec_qIL9ggml_type17ELi1ELb0ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj15HIP_vector_typeIjLj3EEjjjS8_jjjS8_jjjj'
for P in sys.argv[2:]:
    s=open(f'{S}/isa-probe{P}.s').read()
    names=[n for n in re.findall(r'^(_ZL13mul_mat_vec_qIL9ggml_type17ELi1ELb0ELb0ELb0E\S*):',s,re.M)]
    name=names[0]
    body=s.split(name+':',1)[1].split('.Lfunc_end',1)[0]
    ins=[l.split()[0] for l in body.splitlines() if l.startswith('\t') and not l.strip().startswith(('.',';')) and l.split()]
    # find the main loop: the basic block range ending in a backward branch with most instructions
    lines=[l for l in body.splitlines()]
    labels={}; cur=None; blocks=collections.OrderedDict()
    for l in lines:
        m=re.match(r'^(\.LBB\S+):',l)
        if m: cur=m.group(1); blocks[cur]=[]; continue
        if cur and l.startswith('\t') and l.split() and not l.strip().startswith(('.',';')): blocks[cur].append(l.split()[0])
    big=max(blocks.items(), key=lambda kv: len(kv[1]))
    c=collections.Counter(big[1])
    cat=collections.Counter()
    for i,n in c.items():
        if i.startswith('global_load') or i.startswith('buffer_load'): cat['vmem_load']+=n
        elif i.startswith('s_waitcnt'): cat['waitcnt']+=n
        elif i.startswith('v_dot4'): cat['v_dot4']+=n
        elif i.startswith('v_'): cat['valu_other']+=n
        elif i.startswith('s_'): cat['salu/branch']+=n
        elif i.startswith('ds_'): cat['lds']+=n
        else: cat[i]+=n
    m=re.search(re.escape(name)+r'.*?VGPRs: (\d+).*?Occupancy \[waves/SIMD\]: (\d+)',open(f'{S}/resusage-probe{P}.txt').read(),re.S)
    print(f'P={P} loop block {big[0]} n_instr={len(big[1])} {dict(cat)}  VGPR={m.group(1) if m else "?"} occ={m.group(2) if m else "?"}')
    if P=='0':
        print('  top VALU:', [(k,v) for k,v in c.most_common(25) if k.startswith('v_')])
        print('  loads:', [(k,v) for k,v in c.items() if 'load' in k])
