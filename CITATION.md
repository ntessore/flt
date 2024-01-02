The Fast Legendre Transformation employed in this package carries out a FLT by
means of a DCT.  This was first described by Alpert & Rokhlin (1991):

> Alpert, B. K., & Rokhlin, V., 1991, SIAM Journal on Scientific and
> Statistical Computing, 12, 158. doi:10.1137/0912009

The specific implementation transforms between DCT and FLT using a recurrence,
without explicitly constructing the matrix of Alpert & Rokhlin (1991).  For
more information, see:

> Tessore N., Loureiro A., Joachimi B., von Wietersheim-Kramsta M., Jeffrey N.,
> 2023, OJAp, 6, 11. doi:10.21105/astro.2302.01942
