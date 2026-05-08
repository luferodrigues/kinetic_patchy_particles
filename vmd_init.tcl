package require pbctools

mol modstyle 0 top VDW
foreach {elem rad} {
    H 0.7
    He 0.7
    Li 0.7
    Be 0.7
    B 0.7
    C 2.0
    O 2.0
    N 2.0
} {
    set sel [atomselect top "element $elem"]
    $sel set radius $rad
}
mol reanalyze top
pbc set {60 60 60} -all
pbc box -center origin
display projection orthographic
