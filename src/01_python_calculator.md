---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Python as Your Chemistry Calculator

```{admonition} 🟢 Beginner · school & first-year
:class: level-1
No programming experience needed. You only need to know what a mole is.
```

Every chemist starts with a calculator: molar masses, dilutions, pH, gas volumes. A pocket calculator is fine for one answer, but it forgets everything the moment you clear it, it cannot show you its working, and if you mistype Avogadro's constant halfway through a long calculation nothing warns you. **Python** is a programming language, a way of writing instructions that a computer carries out exactly. In this chapter we use it as a calculator that remembers your numbers by name, repeats a calculation for any input you give it, and keeps a written record of every step that you (or a lab partner, or an examiner) can check later.

You do not need to install anything to follow along. Press the 🚀 button at the top of the page and choose **Colab** or **Binder**, and the page opens as a live notebook in your web browser.

## Running your first cell

This page is a **Jupyter notebook**: a document made of **cells**. Text cells, like this paragraph, explain things. **Code cells**, shown in grey boxes, contain Python instructions. To run a code cell, click inside it and press **Shift + Enter**. Python carries out the instructions and shows the result directly underneath, and then the cursor moves to the next cell. You can change the code and run it again as often as you like; nothing can break.

```{code-cell} ipython3
print("Hello, chemistry!")
```

`print` is a **function**: a named, ready-made piece of code that does one job, in this case displaying something on screen. You *call* a function by writing its name followed by round brackets, and whatever you put inside the brackets is the information the function works on. The text in quotation marks is a **string**, Python's word for a piece of text. Quotation marks tell Python "this is text to show, not an instruction to run". Without them Python would go looking for something called `Hello` and complain that it does not exist.

`print` can show several things at once, separated by commas:

```{code-cell} ipython3
print("Water boils at", 100, "°C")
```

## Arithmetic

At its simplest, Python is a calculator. Type a sum in a code cell and run it, and the answer appears. If the last line of a cell is a calculation, Jupyter displays its value even without `print`.

```{code-cell} ipython3
3 + 4
```

Python understands the usual arithmetic **operators** (symbols that combine two numbers). Multiplication is `*` and division is `/`, because the keyboard has no × or ÷. Powers use `**`, because `^` means something else in Python.

| Operator | Meaning | Example | Result |
|---|---|---|---|
| `+` | add | `12.011 + 2 * 15.999` | 44.009 |
| `-` | subtract | `100 - 37.5` | 62.5 |
| `*` | multiply | `2 * 1.008` | 2.016 |
| `/` | divide | `18.0 / 2` | 9.0 |
| `**` | power | `10 ** -3` | 0.001 |
| `//` | floor division (divide, then round down) | `17 // 5` | 3 |
| `%` | remainder after division | `17 % 5` | 2 |

The last two look odd at first but are useful for counting problems. If 17 electrons fill shells two at a time, `17 // 2` gives the 8 complete pairs and `17 % 2` the 1 unpaired electron left over:

```{code-cell} ipython3
print(17 // 2, 17 % 2)
```

The most common calculation in chemistry is a **molar mass**: add up the masses of all the atoms in a formula. Here is water, H₂O, using atomic masses in g mol⁻¹ (H = 1.008, O = 15.999). Two hydrogens and one oxygen:

```{code-cell} ipython3
2 * 1.008 + 15.999
```

And carbon dioxide, CO₂, which has one carbon and two oxygens:

```{code-cell} ipython3
12.011 + 2 * 15.999
```

```{warning}
Python follows the normal order of operations you learned in mathematics: brackets first, then powers, then multiplication and division, then addition and subtraction. `2 * 1.008 + 15.999` is *not* the same as `2 * (1.008 + 15.999)`. The second one doubles the whole sum, which would be the molar mass of two OH radicals. When in doubt, add brackets: they never hurt and they make your intention clear.
```

```{code-cell} ipython3
print(2 * 1.008 + 15.999)      # H2O
print(2 * (1.008 + 15.999))    # 2 × OH: a different calculation
```

## Scientific notation

Chemistry spans enormous ranges, from 10²³ particles in a mole down to 10⁻³⁴ J s for Planck's constant. Typing all those zeros invites mistakes, so Python uses **scientific notation** written with the letter `e`, which you can read as "times ten to the power of". `6.022e23` means 6.022 × 10²³ and `1.6e-19` means 1.6 × 10⁻¹⁹. There must be no spaces inside the number.

```{code-cell} ipython3
6.022e23 * 2        # two moles' worth of particles
```

Python displays very large or very small answers in the same notation, so `1.2044e+24` means 1.2044 × 10²⁴.

## Variables: giving numbers names

Retyping 6.02214076e23 every time you need it is tedious and error-prone. A **variable** solves this: it is a name that refers to a stored value. You create one with a single `=` sign, which in Python means "store the value on the right under the name on the left". It is an *assignment*, not the "equals" of algebra, so read `x = 5` as "x becomes 5".

```{code-cell} ipython3
avogadro = 6.02214076e23   # mol^-1
planck = 6.62607015e-34    # J s
avogadro * planck
```

Anything after a `#` is a **comment**: a note for human readers that Python ignores completely. Use comments to record units and sources, because a number without its unit is the most common mistake in any chemistry calculation, and a comment costs nothing.

Once a variable exists you can use it anywhere you would use the number, and the calculation then reads like the chemistry. Here is the molar mass of ethanol, C₂H₅OH, whose formula contains 2 C, 6 H and 1 O:

```{code-cell} ipython3
m_H = 1.008    # g/mol
m_C = 12.011
m_O = 15.999

M_ethanol = 2 * m_C + 6 * m_H + m_O   # C2H5OH = C2H6O
M_ethanol
```

Variables let us answer a real question in readable steps: **how many moles are in 10.0 g of ethanol?** The amount of substance is mass divided by molar mass,

$$n = \frac{m}{M}$$

```{code-cell} ipython3
mass = 10.0                 # g
moles = mass / M_ethanol
moles
```

And how many molecules is that? Multiply moles by Avogadro's constant, which we stored earlier:

```{code-cell} ipython3
molecules = moles * avogadro
molecules
```

The real power of variables is that you can change an input and re-run. Change `mass = 10.0` to `mass = 25.0`, press Shift + Enter on that cell and the two below it, and every answer updates. A calculator would make you start again from scratch.

A variable can also be updated using its own current value. The right-hand side is worked out first, then stored back under the same name:

```{code-cell} ipython3
volume_mL = 25.0
volume_mL = volume_mL + 0.50      # add 0.50 mL from the burette
volume_mL
```

```{tip}
Good variable names make code read like chemistry: `moles = mass / molar_mass` explains itself, while `c = a / b` does not. Names may contain letters, digits and underscores, cannot start with a digit, and cannot contain spaces. They are case sensitive, so `M` (molar mass) and `m` (mass) are two different variables, a distinction chemists already make on paper.
```

## The `math` module

Python's built-in operators cover arithmetic, but logarithms, square roots, exponentials and trigonometry live in a **module**: a collection of extra functions that you switch on with `import`. You import a module once at the top of a notebook, and after that you use its functions with the module name, a dot, and the function name, for example `math.sqrt(16)`. The dot means "the `sqrt` that belongs to `math`".

```{code-cell} ipython3
import math

print(math.sqrt(16))       # square root
print(math.log10(1000))    # base-10 logarithm
print(math.log(math.e))    # natural logarithm (ln)
print(math.pi)             # the constant π
```

Logarithms are everywhere in chemistry, and the first is pH. pH is defined as $\mathrm{pH} = -\log_{10}[\mathrm{H_3O^+}]$, so a solution with 3.2 × 10⁻⁴ mol L⁻¹ of hydronium has:

```{code-cell} ipython3
conc_H = 3.2e-4            # mol/L of H3O+
pH = -math.log10(conc_H)
pH
```

Exponentials describe how reaction rates depend on temperature. The **Arrhenius equation**, $k = A\,e^{-E_a/RT}$, gives the rate constant $k$ from a pre-exponential factor $A$, an activation energy $E_a$, the gas constant $R$ and the temperature $T$ in kelvin. `math.exp(x)` calculates $e^x$. For a reaction with $E_a = 50$ kJ mol⁻¹ at three temperatures:

```{code-cell} ipython3
A = 1.0e13        # s^-1
Ea = 50_000       # J/mol  (underscores make long numbers readable; Python ignores them)
R = 8.314         # J/(mol K)

for T in (298, 308, 318):
    k = A * math.exp(-Ea / (R * T))
    print(T, "K ->", k, "s^-1")
```

Raising the temperature by just 20 K makes this reaction about three and a half times faster, which is why reactions are so sensitive to heating and why food lasts longer in a fridge. The `for` line repeats the calculation for each temperature in the list. We will learn exactly how it works in Chapter 5.

## Presenting answers: f-strings and significant figures

Python prints every digit it has, such as `0.21706925645807386`, but a long row of digits is not a result. A chemist reports a value with a sensible number of significant figures and a unit. An **f-string** is a string with an `f` in front of the opening quote, and it lets you place values inside text. Anything inside curly brackets `{ }` is calculated and inserted:

```{code-cell} ipython3
print(f"Ethanol molar mass is {M_ethanol} g/mol")
```

After the value you can add a colon and a **format code** that controls how it is displayed. `:.2f` means "2 digits after the decimal point", `:.3e` means "scientific notation with 3 decimals", and `:.3g` means "3 significant figures".

```{code-cell} ipython3
print(f"Molar mass of ethanol: {M_ethanol:.2f} g/mol")
print(f"Moles in {mass} g: {moles:.3f} mol")
print(f"Molecules: {molecules:.3e}")
print(f"pH = {pH:.2f}")
```

| Format | Meaning | `1234.5678` becomes |
|---|---|---|
| `:.2f` | 2 decimal places | `1234.57` |
| `:.3e` | scientific, 3 decimals | `1.235e+03` |
| `:.3g` | 3 significant figures | `1.23e+03` |
| `:,.1f` | thousands separator | `1,234.6` |

Formatting only changes how the number is *displayed*. The variable itself keeps its full precision, so rounding happens once, at the end, exactly as good laboratory practice requires.

## A worked example: the ideal gas law

Let us put everything together. What volume does 2.50 mol of N₂ occupy at 25 °C and 1.00 atm? The ideal gas law $PV = nRT$ rearranges to $V = nRT/P$. We store each input as a named variable, with its unit in a comment, converting to SI units as we go: temperature to kelvin and pressure to pascals.

```{code-cell} ipython3
n = 2.50                 # mol
T = 25 + 273.15          # K
P = 101_325              # Pa (1 atm)
R = 8.314462618          # J/(mol K)

V = n * R * T / P        # m^3
print(f"V = {V:.4f} m³ = {V * 1000:.1f} L")
```

The answer, 61.2 L, is 2.50 × 24.5 L, the familiar molar volume of a gas at room temperature, which is a good sign the units were handled correctly. Checking an answer against a value you already know is a habit worth building from the first day.

```{admonition} ⚗️ Chemistry note
:class: chem-note
Always convert to SI units before using $R = 8.314$ J mol⁻¹ K⁻¹: pressure in pascals, volume in cubic metres, temperature in kelvin. Python does not know what units your numbers are in. It multiplies whatever you give it. In Chapter 18 the `pint` library will attach units to numbers and do this checking for you.
```

```{admonition} Summary
:class: tip
- A notebook is made of cells. Shift + Enter runs a code cell and shows its result underneath.
- Python does arithmetic with `+ - * / **`, follows the usual order of operations, and writes powers of ten as `6.022e23`.
- `name = value` stores a value in a variable. Use clear names and record units in `#` comments.
- `import math` gives `math.log10`, `math.exp`, `math.sqrt` and friends.
- f-strings such as `f"{x:.3g} mol"` report results with sensible significant figures and units.
```

## Exercises

**★ 1.** Calculate the molar mass of glucose, C₆H₁₂O₆.

```{admonition} Solution
:class: dropdown
`6 * 12.011 + 12 * 1.008 + 6 * 15.999` gives 180.156 g/mol.
```

**★★ 2.** A solution of HCl has pH 2.75. Calculate [H₃O⁺] using `10 ** -pH`, and print it to 3 significant figures.

```{admonition} Solution
:class: dropdown
`print(f"{10 ** -2.75:.3g} mol/L")` prints `0.00178 mol/L`.
```

**★★★ 3.** Using the Arrhenius equation, find the temperature at which the rate constant of the reaction above reaches 1.0 s⁻¹. Rearrange to $T = E_a / (R \ln(A/k))$.

```{admonition} Solution
:class: dropdown
`Ea / (R * math.log(A / 1.0))` gives about 201 K.
```

```{admonition} 🤖 ML corner
:class: ml-corner
Machine-learning models are just functions that turn numbers into predictions, exactly like `moles = mass / M`. The difference is that the formula is *learned* from data instead of being written down by a chemist. You will build your first one in Chapter 31.
```
