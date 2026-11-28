# Preface

Chemistry has always been a science of measurement. We weigh, we titrate, we time reactions and we read colours, and then we try to make sense of the numbers. What has changed in recent years is the amount of data. A modern laboratory can produce more readings in an afternoon than a chemist of the past could write down in a lifetime. Spectrometers stream thousands of points a second, databases hold hundreds of millions of structures, and robots run experiments through the night. Chemists now use computers to sort, plot and question all of this as a matter of routine, and, increasingly, to let machine-learning models suggest which molecule or material to make next.

This book is a path into that world, from the very first line of code to the research frontier. It is written for a wide range of readers: school students who like chemistry and are curious about computers, undergraduates who want to analyse their lab data properly, and graduate students and researchers who want to apply cheminformatics and machine learning in their own field. The early chapters assume no programming experience and no mathematics beyond what you meet in secondary school. If you can follow a lab procedure, you can learn to write a program; the two skills are closer than they look. The later chapters build steadily until you are working with the same tools and ideas that appear in today's research papers.

We use Python, a free language that reads almost like English and that scientists use every day, together with the open-source libraries that power modern computational chemistry: NumPy, pandas, Matplotlib, Seaborn, Plotly, SciPy, SymPy, RDKit, py3Dmol and scikit-learn. Each idea is introduced with a familiar piece of chemistry: moles and concentration, pH, gas laws, titrations, reaction rates, periodic trends, spectra, molecular structures and crystals. You will see a great many molecules drawn on the page, functional groups highlighted in colour, reaction schemes, energy diagrams and interactive 3D structures you can rotate with your mouse. Every page is a Jupyter notebook. You can run it, change a number and see what happens. Every output and every graph in this book was produced by running the code shown beside it.

## How the book is organised

The fifty-one chapters are grouped into seven parts.

- **Part I, Python Fundamentals (Chapters 1–11)**, gets you running your first lines of Python and covers the core of the language: numbers and strings, lists and dictionaries, decisions and loops, functions, files, errors, classes and modules. Every example is chemical, from molar masses to a home-made `Molecule` class.
- **Part II, The Scientific Python Stack (Chapters 12–18)**, turns to real data: arrays with NumPy, tables with pandas, figures with Matplotlib, Seaborn and Plotly, fitting and integration with SciPy, exact algebra with SymPy, and honest handling of units and uncertainty.
- **Part III, Cheminformatics and Molecular Visualisation (Chapters 19–24)**, teaches the computer to understand molecules: SMILES and InChI, drawing and describing molecules with RDKit, substructures and fingerprints, reactions in code, 3D conformers and chemical databases.
- **Part IV, Chemistry Applications (Chapters 25–30)**, solves problems from across physical chemistry: kinetics, thermodynamics and equilibrium, spectroscopy, quantum chemistry, molecular simulation and crystallography.
- **Part V, Machine Learning Foundations (Chapters 31–35)**, shows how to train a model, how to represent molecules and materials as numbers, how to map chemical space, how neural networks work, and how Bayesian optimisation chooses the next experiment. Above all, it shows how to tell whether a model is any good.
- **Part VI, Machine Learning Across Chemistry and Materials Science (Chapters 36–46)**, applies these ideas field by field: medicinal, organic and analytical chemistry, catalysis, physical and quantum chemistry, materials, energy materials, polymers, biochemistry and environmental chemistry, generative design, and large language models.
- **Part VII, Research Frontiers (Chapters 47–51)**, looks at AI-assisted discovery, chemical sensing, imaging, predictive process chemistry and trustworthy AI, with references to the open research literature.

Every chapter carries a level badge: 🟢 **beginner** (school and first year), 🟡 **intermediate** (undergraduate) or 🔴 **advanced** (graduate and research). School students can read Part I and the green chapters on their own; university courses can follow Parts I to IV in order; researchers can move quickly to Parts III, V, VI and VII. *How to Use This Book* suggests reading paths for each group.

Each chapter ends with exercises graded ★ (warm-up), ★★ (practice) and ★★★ (challenge), with selected solutions in the appendices. The appendices also contain Python and RDKit cheat sheets, a description of every dataset used, a glossary and the full list of references.

## What you need

A computer with a web browser is enough. The simplest way to start is to click the 🚀 rocket at the top of any chapter and open it in **Google Colab** (colab.research.google.com) or **Binder**, free services that run Python notebooks in the browser. If you would rather work offline, the free **Anaconda** or **Miniforge** distributions install Python, Jupyter and every library used here on your own machine, using the `environment.yml` file that comes with the book. *Installing Python and the Libraries* walks you through both routes. If you are using a family or school computer, ask before installing software.

## A word about the boxes

Shaded boxes in the text pick out things worth pausing on: ⚗️ **chemistry notes** that explain the science behind a calculation, 🤖 **ML corners** that link a topic to machine learning, **tips**, and **warnings** about common mistakes in code and in chemistry. They are there to help, not to be learned by heart.

## A note on sources

All text, code and figures in this book are original. The datasets are either generated in the notebooks themselves or come from openly licensed public sources, which are credited where they are used. Research results are described in my own words and cited in the references, so that you can go to the original papers.

## How this book was made

I used an AI assistant tool (Claude) to draft text, the code examples and the illustrations. If you find a mistake, I would be glad to hear about it, by opening an issue on the book's GitHub page, so that it can be corrected in a future edition.

Most of all, do not just read the code. Type it, break it on purpose, read the error messages and fix them. That is how every programmer learned, and it is the quickest way to make these ideas your own.

*Dr. Bahadur Sk*<br>
*Koneru Lakshmaiah Education Foundation (KL Deemed to be University), 2026*
