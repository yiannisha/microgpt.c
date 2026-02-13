#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>
#include <math.h>

// --- Dataset ---

typedef char** Dataset;

size_t load_dataset(Dataset *dataset, const char *dataset_filepath) {
    FILE *f = fopen(dataset_filepath, "r");
    if (!f) {
        // handle missing file error
        printf("File %s not found!\n", dataset_filepath);
        exit(EXIT_FAILURE);
    }
    
    Dataset lines = NULL;
    char *line = NULL;
    size_t count = 0;
    size_t n = 0;

    while (getline(&line, &n, f) != -1) {
        lines = realloc(lines, (count+1) * sizeof(char *));
        lines[count] = strdup(line);
        count++;
    }

    *dataset = lines;

    free(line);
    fclose(f);

    return count;
}

void shuffle_dataset(Dataset dataset, size_t n) {
    if (n <= 1) return;
    size_t i;
    for (i=0; i<n; ++i) {
        size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
        char *t = dataset[j];
        dataset[j] = dataset[i];
        dataset[i] = t;
    }
}

// --- Tokenizer ---

typedef struct {
    char* items;
    size_t size;
} Tokenizer;

Tokenizer tokenizer_init() {
    // char-level tokenization, so we only need ASCII characters
    char *items = malloc(128 * sizeof(char));
    for (int i=0; i<128; ++i) items[i] = (char)-1;
    return (Tokenizer){items, 0};
}

void tokenizer_insert(Tokenizer *t, char c) {
    if (c < 20 || c > 126) return; // keep only "normal" characters
    if (t->items[(int)c] >= 0) return; // token id already allocated
    t->items[(int)c] = t->size++;
}

// --- Autograd ---
typedef struct Value {
    double data;
    struct Value** children;
    double* local_grads;
    double grad;

    // readonly metadata
    size_t _children_num;
} Value;

Value val_from_const(double a) { return (Value){ a, NULL, NULL, 0.0, 0 }; }
Value val_init(double data, int children_num) {
    Value out = {data};
    out.grad = 0.0;
    out._children_num = children_num;

    out.children = malloc(out._children_num * sizeof(*out.children));
    out.local_grads = malloc(out._children_num * sizeof(*out.local_grads));

    if (!out.children || !out.children) {
        perror("Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    return out;
}

// --- Custom Operators for `Value` objects ---

// TODO: these can be minified (single-line)
Value _add(Value *a, Value *b) {
    Value out = val_init(a->data + b->data, 2);

    out.children[0] = a;
    out.children[1] = b;
    out.local_grads[0] = 1.0; // d(a+b) / da
    out.local_grads[1] = 1.0; // d(a+b) / db

    return out;
}

Value _mul(Value *a, Value *b) {
    Value out = val_init(a->data * b->data, 2);

    out.children[0] = a;
    out.children[1] = b;
    out.local_grads[0] = b->data; // d(a*b) / da
    out.local_grads[1] = a->data; // d(a*b) / db

    return out;
}

Value _pow(Value *a, int pwr) {
    double acc = 1.0;
    for (int i=0; i<pwr-1; ++i) acc *= a->data;
    
    Value out = val_init(acc * a->data, 1);

    out.children[0] = a;
    out.local_grads[0] = pwr * acc; // d(a**pwr) / da

    return out;
}

Value _log(Value *a) {
    Value out = val_init(log(a->data), 1);
    
    out.children[0] = a;
    out.local_grads[0] = 1 / a->data; // d(log(a)) / da

    return out;
}

Value _exp(Value *a) {
    Value out = val_init(exp(a->data), 1);
    
    out.children[0] = a;
    out.local_grads[0] = exp(a->data); // d(exp(a)) / da

    return out;
}

Value _relu(Value *a) {
    Value out = val_init(fmax(a->data, 0), 1);

    out.children[0] = a;
    out.local_grads[0] = a->data > 0 ? 1.0 : 0.0;

    return out;
}

Value _neg(Value a) {
    Value out = a;
    out.data = -out.data;

    return out;
}

Value _sub(Value *a, Value *b) {}
Value _div(Value *a, Value *b) {}

void print_val(Value *a) {
    // parse string of addresses to children
    char *buf = calloc(1000, sizeof(char));
    size_t s = 0;

    for (size_t i=0; i<a->_children_num; ++i) {
        s += sprintf(buf+s, "%p: [ data: %f ], ", a->children[i], a->children[i]->data);
    }

    printf("Value( data=%f, children={ %s }, grad=%f, address=%p )\n", a->data, buf, a->grad, a);
}

// --- Main Training/Inference Loop ---

int main() {
    srand( time(NULL) );

    // 1. retrieve dataset file if it doesn't exist 
    const char *path = "input.txt";
    Dataset dataset = NULL;
    size_t size = load_dataset(&dataset, path);

    // 2. shuffle dataset
    shuffle_dataset(dataset, size);

    // 3. create a simple tokenizer
    Tokenizer t = tokenizer_init();
    char *c_ptr = NULL;
    for (size_t i=0; i<size; ++i) {
        c_ptr = dataset[i];
        while (*c_ptr != '\0') {
            tokenizer_insert(&t, *c_ptr);
            c_ptr++;
        }
    }

    // debug
    // for (int i=0; i<128; i++) printf("%d ", t.items[i]);
    // printf("\n");

    Value a = val_from_const(3);
    printf("a: ");
    print_val(&a);
    Value b = val_from_const(4);
    printf("b: ");
    print_val(&b);
 
    Value c = _mul(&a, &b);
    printf("c: ");
    print_val(&c);

    Value d = _add(&b, &a);
    printf("d: ");
    print_val(&d);

    Value n = _neg(a);
    printf("n: ");
    print_val(&n);

    Value f = _add(&b, &n);
    printf("f: ");
    print_val(&f);

    Value n2 = _neg(f);
    printf("n2: ");
    print_val(&n2);

    Value g = _add(&f, &n2);
    printf("g: ");
    print_val(&g);

    free(dataset);

    return EXIT_SUCCESS;
}