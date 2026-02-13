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

// --- Custom Operators for `Value` objects ---

// TODO: these can be minified (single-line)
Value _add(Value *a, Value *b) {
    Value *children[] = { a, b };
    double local_grads[] = { 1.0, 1.0 };

    Value local = (Value){
        (*a).data + (*b).data, children, local_grads, 0.0, 2
    };

    return local;
}

Value _mul(Value *a, Value *b) {
    Value *children[] = { a, b };
    double local_grads[] = { a->data, b->data };

    Value local = (Value){
        a->data * b->data, children, local_grads, 0.0, 2
    };

    // TODO: if I comment this line it breaks, the addresses changes wtf? does this make not copy (which is what I want to do)?
    printf("_mul: children[0]: %p, children[1]: %p\n", local.children[0], local.children[1]);

    return local;
}

Value _pow(Value *a, unsigned int pwr) {
    double acc = 1.0;
    for (unsigned int i=0; i<pwr-1; ++i) acc *= a->data;
    Value *children[] = { a };
    double local_grads[] = { pwr * acc };
    Value local = (Value){
        acc * a->data, children, local_grads, 0.0, 1
    };

    return local;
}

Value _log(Value *a) {
    Value *children[] = { a };
    double local_grads[] = { 1 / a->data };
    
    Value local = (Value){
        log(a->data), children, local_grads, 0.0, 1
    };

    return local;
}

Value _exp(Value *a) {
    Value *children[] = { a };
    double e = exp(a->data);
    double local_grads[] = { e };
    
    Value local = (Value){
        e, children, local_grads, 0.0, 1
    };

    return local;
}

Value _relu(Value *a) {
    Value *children[] = { a };
    double local_grads[] = { a->data > 0 ? 1.0 : 0.0 };

    Value local = (Value){
        fmax(a->data, 0), children, local_grads, 0.0, 1
    };

    return local;
}

Value _neg(Value *a) {
    Value v = val_from_const(-1.0);
    return _mul(a, &v);
}

Value _sub(Value *a, Value *b) {
    printf("_sub: a: %p, b: %p\n", a, b);
    Value n = _neg(b);
    printf("_sub: n: %p\n", n);
    return _add(a, &n);
}

Value _div(Value *a, Value *b) {
    Value d = _pow(b, -1.0);
    return _mul(a, &d);
}

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
        while (*(c_ptr++) != '\0') tokenizer_insert(&t, *c_ptr);
    }

    // debug
    // for (int i=0; i<128; i++) printf("%d ", t.items[i]);
    // printf("\n");

    Value a = val_from_const(3);
    print_val(&a);
    Value b = val_from_const(4);
    print_val(&b);
 
    Value c = _mul(&a, &b);
    print_val(&c);

    Value d = _add(&b, &a);
    print_val(&d);

    Value f = _sub(&b, &a);
    print_val(&f);

    free(dataset);

    return EXIT_SUCCESS;
}