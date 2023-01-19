def RK4(Fx,Gx,Fy,Gy,x,vx,y,vy,h):
    """ Runge-Kutta, 4rd order, 2D, fills arrays x,y,vx,vy,t. Use: RK2(F,G,t,x,v,h,N)
    for time t,x,y,vx,vy N-arrays, x'=Fx(x,y,vx,vy,t), vx'=Gx(x,y,vx,vy,t),
    y'=Fy(x,y,vx,vy,t), vy'=Gy(x,y,vx,vy,t),
    t,x,y,vx,vy should be all zeros except first element, which are the initial 
    values"""   
    k1x=Fx(x,y,vx,vy)*h
    l1x=Gx(x,y,vx,vy)*h
    k1y=Fy(x,y,vx,vy)*h
    l1y=Gy(x,y,vx,vy)*h       

    k2x=Fx(x+k1x/2.0,y+k1y/2.0,vx+l1x/2.0,vy+l1y/2.0)*h
    l2x=Gx(x+k1x/2.0,y+k1y/2.0,vx+l1x/2.0,vy+l1y/2.0)*h
    k2y=Fy(x+k1x/2.0,y+k1y/2.0,vx+l1x/2.0,vy+l1y/2.0)*h
    l2y=Gy(x+k1x/2.0,y+k1y/2.0,vx+l1x/2.0,vy+l1y/2.0)*h

    k3x=Fx(x+k2x/2.0,y+k2y/2.0,vx+l2x/2.0,vy+l2y/2.0)*h
    l3x=Gx(x+k2x/2.0,y+k2y/2.0,vx+l2x/2.0,vy+l2y/2.0)*h
    k3y=Fy(x+k2x/2.0,y+k2y/2.0,vx+l2x/2.0,vy+l2y/2.0)*h
    l3y=Gy(x+k2x/2.0,y+k2y/2.0,vx+l2x/2.0,vy+l2y/2.0)*h

    k4x=Fx(x+k3x,y+k3y,vx+l3x,vy+l3y)*h
    l4x=Gx(x+k3x,y+k3y,vx+l3x,vy+l3y)*h
    k4y=Fy(x+k3x,y+k3y,vx+l3x,vy+l3y)*h
    l4y=Gy(x+k3x,y+k3y,vx+l3x,vy+l3y)*h        

    x = x+(k1x+2.0*k2x+2.0*k3x+k4x)/6.0
    vx = vx+(l1x+2.0*l2x+2.0*l3x+l4x)/6.0
    y = y+(k1y+2.0*k2y+2.0*k3y+k4y)/6.0
    vy = vy+(l1y+2.0*l2y+2.0*l3y+l4y)/6.0
    
    return x, y, vx, vy