//> using scala "3.3"
//> using dep "ch.unibas.cs.gravis::scalismo-ui:0.92.0"
// !!! if you are working on a Mac with M1 or M2 processor, use the following import instead !!!
// //> using dep "ch.unibas.cs.gravis::scalismo-ui:0.92,exclude=ch.unibas.cs.gravis%vtkjavanativesmacosimpl"


import scalismo.ui.api.ScalismoUI

object HelloScalismo extends App {
    ScalismoUI()
}