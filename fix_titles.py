import sqlite3, base64, os, time

DB = os.path.expandvars(r"%APPDATA%\antigravity\User\globalStorage\state.vscdb")
BRAIN = os.path.expandvars(r"%USERPROFILE%\.gemini\antigravity\brain")
CONVS = os.path.expandvars(r"%USERPROFILE%\.gemini\antigravity\conversations")

titles = {}
for cid in (f[:-3] for f in os.listdir(CONVS) if f.endswith('.pb')):
    bp = os.path.join(BRAIN, cid)
    if not os.path.exists(bp): continue
    for item in os.listdir(bp):
        if item.startswith('.') or not item.endswith('.md'): continue
        with open(os.path.join(bp, item), 'r', encoding='utf-8', errors='replace') as f:
            line = f.readline().strip()
        if line.startswith('#'): titles[cid] = line.lstrip('# ')[:80]
        break

def ev(v):
    r=b""
    while v>0x7F:r+=bytes([(v&0x7F)|0x80]);v>>=7
    r+=bytes([v&0x7F]);return r or b'\x00'
def dv(d,p):
    r,s=0,0
    while p<len(d):
        b=d[p];r|=(b&0x7F)<<s
        if(b&0x80)==0:return r,p+1
        s+=7;p+=1
    return r,p
def esf(fn,s):
    b=s.encode('utf-8');return ev((fn<<3)|2)+ev(len(b))+b
def ebf(fn,b):
    return ev((fn<<3)|2)+ev(len(b))+b

conn=sqlite3.connect(DB);cur=conn.cursor()
cur.execute("SELECT value FROM ItemTable WHERE key='antigravityUnifiedStateSync.trajectorySummaries'")
decoded=base64.b64decode(cur.fetchone()[0])

entries,p=[],0
while p<len(decoded):
    t,np=dv(decoded,p)
    if(t>>3)!=1 or(t&7)!=2:break
    l,np=dv(decoded,np);entries.append(decoded[np:np+l]);p=np+l

rebuilt=b""
for entry in entries:
    ep,uid,ib=0,None,None
    while ep<len(entry):
        t,ep=dv(entry,ep)
        if(t&7)==2:
            l,ep=dv(entry,ep);c=entry[ep:ep+l];ep+=l
            if(t>>3)==1:uid=c.decode()
            elif(t>>3)==2:
                ip=0;it,ip=dv(c,ip);il,ip=dv(c,ip);ib=c[ip:ip+il].decode()
        elif(t&7)==0:_,ep=dv(entry,ep)
    if uid and ib:
        title=titles.get(uid)
        if not title:
            cp=os.path.join(CONVS,f"{uid}.pb")
            d=time.strftime("%b %d",time.localtime(os.path.getmtime(cp))) if os.path.exists(cp) else "?"
            title=f"Conversation ({d}) {uid[:8]}"
        idata=base64.b64decode(ib);fields,fp=[],0
        while fp<len(idata):
            sp=fp;t,fp=dv(idata,fp);fn,wt=t>>3,t&7
            if wt==0:_,fp=dv(idata,fp)
            elif wt==2:l,fp=dv(idata,fp);fp+=l
            elif wt==1:fp+=8
            elif wt==5:fp+=4
            else:break
            fields.append((fn,wt,idata[sp:fp]))
        nr=b""
        for fn,wt,raw in fields:
            nr+=esf(1,title) if fn==1 and wt==2 else raw
        ne=esf(1,uid)+ebf(2,esf(1,base64.b64encode(nr).decode()))
        rebuilt+=ebf(1,ne)
    else:rebuilt+=ebf(1,entry)

cur.execute("UPDATE ItemTable SET value=? WHERE key='antigravityUnifiedStateSync.trajectorySummaries'",(base64.b64encode(rebuilt).decode(),))
conn.commit();conn.close()
print(f"Fixed {len(titles)} real titles. Open Antigravity now.")