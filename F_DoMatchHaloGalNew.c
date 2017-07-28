//#include "~/Projects/cone/ReadCone/i/Cat/F_OrientationGalaxies.i"
require,"octree.i";

func MatchHalGal(Gal,Hal,idigal)
{
  /*
    DOCUMENT:
    if idigal=1, return the tab containing the ids of the halo of level 1 corresponding to the galaxies;
    if idigal>1, returns the tab containing the ids of the halo of the max level corresponding to the galaxy
    if idigal=0, return the id of the central galaxy contained in the halo.
  */

  "ok";
  posHal=transpose([Hal.posx,Hal.posy,Hal.posz]);//*Hal.aexp(-,);
  posGal=Gal.posc;//*Gal.aexp(-,);
  if (idigal==1)
    {
      w=where((Hal.level==1));
      Hal1=Hal(w);
      posHal1=posHal(,w);//*Hal1.aexp(-,);
      trr=BuildOctreeFromPos(posHal1);
      id=FindNeighbours(trr,posGal,1,d,dist=1);
      ww=where(d>1.5*Hal1(id).rad/(1/(Hal1(id).z+1)));
      numberof(ww);
      idd=w(id);
      //idd(ww)=0;
      return idd;
      
    }
  else if (idigal==0)
    {
      trr=BuildOctreeFromPos(posGal);
      id=FindNeighbours(trr,posHal,1,d,dist=1);
      ww=where(d>1.5*Hal.rad/Hal.aexp);
      id(ww)=0;
      return id;      
    }
  else
    {
      dd=array(999.,5,numberof(posGal(1,)));
      idd=array(0,5,numberof(posGal(1,)));
      for (i=1;i<=5;i++)
        {
          w=where((Hal.level==i));
          if (numberof(w)>0)
            {
          Halt=Hal(w);
          posHalt=posHal(,w);//*Halt.aexp(-,);
          if (numberof(Halt)>300)
            {
          trr=BuildOctreeFromPos(posHalt);
          
          id=FindNeighbours(trr,posGal,1,d,dist=1);
          dd(i,:)=d;
          idd(i,)=w(id);
            }
          else
            {
              dist=array(0.,numberof(Halt),numberof(posGal(1,)));
              for (j=1;j<=numberof(Halt);j++)
                {
                  
                  dist(j,)=sqrt((posHalt(1,j)-posGal(1,))^2+(posHalt(2,j)-posGal(2,))^2+(posHalt(3,j)-posGal(3,))^2);
                
                }
              for (j=1;j<=numberof(Gal);j++)
                {
                  dd(i,j)=min(dist(,j));
                  idd(i,j)=where(dist(,j)==min(dist(,j)))(1);
                }
            }
            }
        }
      iddd=array(0,numberof(Gal));
      stat,dd;
      for (i=1;i<=numberof(Gal);i++)
        {
          w=where(dd(,i)==min(dd(,i)));
          iddd(i)=idd(w(1),i);
        }
      return iddd;    
    }
  
}


func MatchedCatalog(Hal,Gal,outname)
{
  
  ngal=numberof(Gal);
  nhal=numberof(Hal);
  tabgal=array(0.,4,ngal);
  tabhal=array(0.,2,nhal);
  tabhal(1,)=span(1,nhal,nhal);
  tabhal(2,)=MatchHalGal(Gal,Hal,0);
  tabgal(1,)=span(1,ngal,ngal);
  tabgal(2,)=log10(Gal.mass*10.^11);

  tabgal(3,)=MatchHalGal(Gal,Hal,1);
  tabgal(4,)=MatchHalGal(Gal,Hal,2);
  smwrite(outname+"_gal.txt",transpose(tabgal));
  smwrite(outname+"_hal.txt",transpose(tabgal));
}
