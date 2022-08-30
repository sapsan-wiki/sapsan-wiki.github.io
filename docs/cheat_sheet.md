---
search:
  exclude: true

tags:
  - HTML5
  - JavaScript
  - CSS
...

# Cheat-Sheet

`mike set-default --push latest`
`mike deploy --push --update-aliases 0.1 latest`
: publishes a new wiki version

[Hover me](https://example.com "I'm a tooltip!")

:material-information-outline:{title="Important information"}

GLSR

*[GLSR]: abbreviation glossary in includes/abbreviations.md

<!---  12 types: [note, info, tldr, tip, success, help, warning, fail, danger, bug, example, quote]
       and you can do custom icons/colors as well --->

!!! example "Admonition"
    smth here

!!! bug ""

    admonition without a title

??? note "Collapsible"

    Oh what! You clicked?!

???+ tldr "Collapsible"

    Already expanded    

!!! info inline end

    inline or inline end to inline admonition with text

Lorem ipsum dolor sit amet, consectetur
adipiscing elit. Nulla et euismod nulla.
Curabitur feugiat, tortor non consequat
finibus, justo purus auctor massa, nec
semper lorem quam in massa.

Lorem footnote[^1] dolor sit amet, consectetur adipiscing another one[^2]
[^1]: Lorem ipsum dolor sit amet, consectetur adipiscing elit.
[^2]:
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla et euismod
    nulla. Curabitur feugiat, tortor non consequat finibus, justo purus auctor
    massa, nec semper lorem quam in massa.

1.  Vivamus id mi enim. Integer id turpis sapien. Ut condimentum lobortis

    1.  Vivamus venenatis porttitor tortor sit amet rutrum. Pellentesque aliquet        

    2.  Morbi eget dapibus felis. Vivamus venenatis porttitor tortor sit amet        

        1.  Mauris dictum mi lacus
        2.  Ut sit amet placerat ante    

- [x] Checked box
- [ ] unchecked box
    * [x] In hac habitasse platea dictumst    

| Method      | Description                          |
| ----------- | ------------------------------------ |
| `GET`       | :material-check:     Fetch resource  |
| `PUT`       | :material-check-all: Update resource |
| `DELETE`    | :material-close:     Delete resource |

$$
\operatorname{ker} f=\{g\in G:f(g)=e_{H}\}{\mbox{.}}
$$

Also works as in latex inline $19 M_{\odot}$

``` yaml
theme:
  features:
    - content.code.annotate # (1)!
```

1.  :man_raising_hand: I'm a code annotation! I can contain `code`, __formatted
    text__, images, ... basically anything that can be written in Markdown.

``` py title="bubble_sort.py" hl_lines="2 3"
def bubble_sort(items):
    for i in range(len(items)):
        for j in range(len(items) - 1 - i):
            if items[j] > items[j + 1]:
                items[j], items[j + 1] = items[j + 1], items[j]
```    
The `#!python range()` function is used to generate a sequence of numbers.

 <font size="1"> This is my text number 1</font>

 <font size="2"> This is my text number 2</font>

 <font size="3"> This is my text number 3</font> 

 <font size="4"> This is my text number 4</font> 

 <font size="5"> This is my text number 5</font> 

 <font size="6"> This is my text number 6</font>

 <span style="color:rgb(103, 145, 224)">some *blue* text</span>.

 <span style="color:var(--class-color)">some *CLASS* text</span>.


<script> 
  var hcolor = 'hi'
  document.getElementById('output').innerHTML = hcolor;
</script>

<body>
    <p id="output"></p>
    <p style="color: var(--my-color)"></p>
</body>
