---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Numbers, Booleans and Types

```{admonition} 🟢 Beginner · school & first-year
:class: level-1
Builds on Chapter 1.
```

Every value in Python has a **type**, a label that tells Python what kind of thing the value is and therefore what you are allowed to do with it. You can divide one mass by another, but dividing a mass by the word "carbon" makes no sense, and Python refuses because the two values have incompatible types. Knowing the type of your data saves you from many silent mistakes: a concentration read from a file as text looks like a number on screen, yet it will not behave like one until you convert it. This chapter introduces the handful of basic types you will use in every later chapter, shows how to check and convert them, and explains the one surprise that catches every scientist at least once: floating-point numbers are not exact.

## The built-in numeric types

Python has three kinds of number and one kind of truth value built in. An **`int`** (integer) is a whole number with no decimal point, such as 6 or −2. A **`float`** (floating-point number) is a real number written with a decimal point or in scientific notation, such as 12.011 or 6.022e23. A **`complex`** number has a real and an imaginary part, written with `j` for the imaginary unit. A **`bool`** (Boolean) is either `True` or `False`. Python decides the type from the way you write the value, so `6` is an int while `6.0` is a float, even though they are equal as numbers.

The built-in function `type()` tells you the type of any value. This is the first thing to try whenever a calculation behaves unexpectedly.

```{code-cell} ipython3
print(type(6))
print(type(6.0))
print(type("6"))
```

The three results differ, which is exactly why `"6"` (the character six, a piece of text) cannot be multiplied by a mass. Here is the same idea with values that mean something chemically:

```{code-cell} ipython3
n_atoms = 6              # int: whole numbers, counts of things
mass = 12.011            # float: real numbers with a decimal point
wave = 3 + 4j            # complex: used in quantum mechanics and Fourier transforms
is_metal = False         # bool: True or False

for value in (n_atoms, mass, wave, is_metal):
    print(repr(value), "is", type(value).__name__)
```

The `for` loop simply repeats the `print` line once for each value; loops are covered properly in Chapter 5. The function `repr()` shows a value the way you would type it, and `.__name__` gives the type's short name.

```{admonition} ⚗️ Chemistry note
:class: chem-note
Use `int` for things you count (atoms, electrons, protons) and `float` for things you measure (masses, energies, concentrations). Counting 6.0 electrons does not make sense; measuring exactly 12 g does not happen.
```

Arithmetic that mixes an int and a float gives a float, because Python promotes the result to the type that can hold more information. Adding 2 hydrogen atoms' worth of mass to an integer count therefore never loses the decimals:

```{code-cell} ipython3
n_H = 2                  # int
m_H = 1.008              # float
print(n_H * m_H, type(n_H * m_H).__name__)
```

## Converting between types

Converting means making a new value of a different type from an existing one, and you do it by calling the type's name as a function: `float(x)`, `int(x)`, `str(x)`. You need conversion constantly, because data rarely arrives in the type you want. Anything typed by a user with `input()`, read from a text file, or pulled from a spreadsheet column often starts life as text, and text must become a number before you can do arithmetic with it.

The simplest case is turning a piece of text into a float:

```{code-cell} ipython3
reading = "0.0253"               # text typed by a user or read from a file
print(type(reading).__name__)
conc = float(reading)            # now a number
print(type(conc).__name__, conc * 1000, "mmol/L")
```

Conversion between number types follows rules that are worth knowing precisely. `int()` throws away the decimal part (it truncates towards zero) rather than rounding, so `int(7.9)` is 7. The `round()` function rounds to the nearest whole number, and when a value lies exactly halfway it rounds to the nearest *even* number. This "banker's rounding" avoids a systematic upward bias when many values are rounded, which matters when you average thousands of readings.

```{code-cell} ipython3
print(int(7.9))                  # int() truncates towards zero
print(round(7.9), round(2.5), round(3.5))   # round() uses banker's rounding
print(round(18.01528, 2))        # round to two decimal places
print(str(18.015) + " g/mol")    # str() turns a number into text
```

Conversion fails loudly when the text is not a valid number. That is a good thing: it is far better to get an error than a wrong answer.

```{code-cell} ipython3
try:
    float("0.025 M")
except ValueError as err:
    print("ValueError:", err)
```

The `try` / `except` structure catches the error so the notebook can continue; Chapter 9 explains it in detail. For now, notice that the unit "M" stuck to the number is enough to stop the conversion, so strip units off before converting.

## Integer and float division

Python has two division operators because there are two different questions you might be asking. The single slash `/` is *true division* and always returns a float, even when the answer is whole. The double slash `//` is *floor division*: it returns how many whole times one number fits into another, rounded down. The percent sign `%` is the *modulo* operator and returns the remainder left over after floor division. Together, `//` and `%` split a number into "complete groups" and "what is left".

```{code-cell} ipython3
print(10 / 2)    # 5.0, a float even though the answer is whole
print(10 // 3)   # 3 complete groups of 3
print(10 % 3)    # 1 left over
```

Electron counting is a natural use. Seventeen electrons (a chlorine atom) form eight complete pairs with one electron left unpaired:

```{code-cell} ipython3
electrons = 17
print(electrons / 2)    # always a float
print(electrons // 2)   # electron pairs
print(electrons % 2)    # unpaired electrons left over
```

The `%` operator answers a chemistry question directly: an odd electron count means a **radical**. The test `n % 2 == 1` is the standard way to ask "is this number odd?".

```{code-cell} ipython3
for species, n_e in [("NO", 15), ("CO", 14), ("O2", 16), ("CH3", 9)]:
    print(species, "radical" if n_e % 2 == 1 else "even-electron")
```

O₂ has an even electron count yet is famously a diradical, a reminder that simple parity tells you about odd electrons only, not about every unpaired one.

## Floating-point precision

A float is stored in the computer's memory as a binary fraction with a fixed number of digits, about 15 to 17 significant decimal digits. Just as 1/3 cannot be written exactly in decimal (0.3333… goes on forever), most decimal fractions such as 0.1 cannot be written exactly in binary. Python stores the closest binary value it can, and the tiny difference shows up when you do arithmetic.

```{code-cell} ipython3
print(0.1 + 0.2)
print(0.1 + 0.2 == 0.3)
```

The error is about one part in 10¹⁶, far smaller than any real measurement uncertainty, so it never matters for the *size* of a chemical result. It matters only when you test two floats for exact equality, because "almost equal" is not "equal". The rule is therefore simple: never compare floats with `==`. Use `math.isclose`, which asks whether two numbers agree within a tolerance you choose. The argument `rel_tol` is a relative tolerance, so `rel_tol=1e-3` means "within 0.1 %".

```{code-cell} ipython3
import math
print(math.isclose(0.1 + 0.2, 0.3))

measured, literature = 1.0012, 1.0010     # g/mL
print(math.isclose(measured, literature, rel_tol=1e-3))   # within 0.1 %?
```

This is also how you should check a result against a literature value in your own code: decide what agreement is acceptable for the experiment, then test against that tolerance.

## Comparisons and Booleans

A **comparison** asks a yes/no question about two values and answers with a Boolean, `True` or `False`. Booleans are what let a program make decisions (Chapter 5), filter data (Chapters 12 and 13) and flag problems. The comparison operators look like their mathematical symbols, except that equality uses a double `==`, because a single `=` already means "store a value in a variable".

| Operator | Meaning |
|---|---|
| `==`, `!=` | equal, not equal |
| `<`, `<=`, `>`, `>=` | ordering |
| `and`, `or`, `not` | combine Booleans |

A single comparison is the simplest Boolean expression:

```{code-cell} ipython3
pH = 4.2
print(pH < 7)
print(pH == 7)
```

Booleans combine with `and` (both must be true), `or` (at least one must be true) and `not` (flips the value). Python also allows *chained* comparisons such as `4 < pH < 7`, which reads exactly like the mathematics and means `4 < pH and pH < 7`.

```{code-cell} ipython3
print("acidic:", pH < 7)
print("weakly acidic:", 4 < pH < 7)    # chained comparison

boiling_point = 78.4   # °C, ethanol
melting_point = -114   # °C, ethanol
room_temp = 25
print("liquid at room temperature:", melting_point < room_temp < boiling_point)
print("not a gas:", not room_temp > boiling_point)
```

Booleans are secretly numbers: `True` behaves as 1 and `False` as 0. This lets you count how many conditions are true with `sum()`, a trick used constantly in data analysis.

```{code-cell} ipython3
pH_values = [3.1, 7.4, 6.8, 2.2, 9.0]
print("acidic samples:", sum(p < 7 for p in pH_values))
```

## Complex numbers

A complex number has the form $a + bi$, where $i = \sqrt{-1}$; Python writes the imaginary unit as `j`, as engineers do. You may not need complex numbers in a first chemistry course, but they appear in wavefunctions, which carry a phase, and in the Fourier transform that every NMR and FTIR spectrometer uses to turn a time signal into a spectrum. Python handles them with the same operators as other numbers, and the `cmath` module provides complex versions of `exp`, `sqrt` and friends.

```{code-cell} ipython3
z = 3 + 4j
print(z.real, z.imag)       # the two parts
print(abs(z))               # the magnitude, sqrt(3² + 4²) = 5
```

A quantum-mechanical phase factor $e^{i\theta}$ has magnitude 1, so it changes a wavefunction's phase but not the probability $|\psi|^2$ of finding the particle:

```{code-cell} ipython3
import cmath
psi = cmath.exp(1j * math.pi / 3)      # a phase factor e^{iθ}
print(psi)
print("probability |ψ|² =", abs(psi) ** 2)
```

## Very large and very small numbers

Integers in Python have unlimited size; the only limit is your computer's memory, so you can compute exact factorials or permutation counts of any size. Floats are different. They have a fixed range, from about 10⁻³⁰⁸ to 10³⁰⁸. A result that is too large *overflows* to the special value `inf` (infinity), and one too close to zero *underflows* to 0.0.

```{code-cell} ipython3
print(2 ** 200)                 # ints have unlimited size
print(1e308 * 10)               # floats overflow to infinity
print(5e-324 / 2)               # and underflow to zero
```

The number of ways to arrange 20 different atoms in a row is an exact integer, which shows the practical value of unlimited ints:

```{code-cell} ipython3
print(math.factorial(20))
```

```{tip}
Equilibrium constants can be enormous (K ≈ 10⁸⁰). Work with $\log K$ or $\Delta G$ instead to stay within float range. Chemists already do this with pH and pK<sub>a</sub>, which are logarithms for exactly the same reason.
```

```{admonition} Summary
:class: tip
- Every value has a type: `int` for counts, `float` for measurements, `complex` for phases, `bool` for yes/no. Check it with `type()`.
- Convert with `int()`, `float()` and `str()`; text from files and users usually needs converting before arithmetic.
- `/` always gives a float, `//` gives whole groups and `%` the remainder.
- Floats are approximate, so compare them with `math.isclose`, never `==`.
- Comparisons give Booleans, which combine with `and`, `or`, `not` and can be counted with `sum()`.
- Floats overflow above about 10³⁰⁸; use logarithms for huge quantities.
```

## Exercises

**★ 1.** An element has 26 protons and forms an ion with 23 electrons. Use integers to compute its charge and print `Fe3+`.

```{admonition} Solution
:class: dropdown
`charge = 26 - 23; print(f"Fe{charge}+")`
```

**★★ 2.** Write a Boolean expression that is `True` if a temperature `T` in kelvin lies between the melting point (273.15 K) and boiling point (373.15 K) of water.

```{admonition} Solution
:class: dropdown
`273.15 < T < 373.15`
```

**★★★ 3.** Compute $K = e^{-\Delta G / RT}$ for ΔG = −500 kJ mol⁻¹ at 298 K. What happens? Compute $\log_{10} K$ instead.

```{admonition} Solution
:class: dropdown
`math.exp(500e3/(8.314*298))` raises `OverflowError`. `500e3/(8.314*298)/math.log(10)` gives log K ≈ 87.6.
```
