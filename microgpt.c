#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- Random Number Generation ---

double drand() { return (rand() + 1.0) / (RAND_MAX + 1.0); }

double random_normal() { return sqrt(-2*log(drand())) * cos(2 * M_PI * drand() ); }

double random_gaussian(double mean, double std) { return mean + std * random_normal(); }

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
    if (c < 20 || c > 126) return; // keep only "normal" ASCII characters
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
    unsigned short int _visited;
} Value;

Value val_from_const(double a) { return (Value){ a, NULL, NULL, 0.0, 0, 0 }; }
Value val_init(double data, int children_num) {
    Value out = {data};
    out.grad = 0.0;
    out._children_num = children_num;
    out._visited = 0;

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

// TODO: minify using _add and _neg
Value _sub(Value *a, Value *b) {
    Value out = val_init(a->data - b->data, 2);

    out.children[0] = a;
    out.children[1] = b;
    out.local_grads[0] = 1.0; // d(a-b) / da
    out.local_grads[1] = -1.0; // d(a-b) / db

    return out;
}

// TODO: minify using _mul and _neg
Value _div(Value *a, Value *b) {
    if (!b->data) {
        perror("Division with zero.\n");
        exit(EXIT_FAILURE);
    }

    Value out = val_init(a->data / b->data, 2);

    out.children[0] = a;
    out.children[1] = b;
    out.local_grads[0] = 1 / b->data; // d(a/b) / da
    out.local_grads[1] =  - a->data / (b->data * b->data); // d(a/b) / db

    return out;
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

// --- Topological Ordering & Backward ---

void build_topo(Value *a, Value ***topo, int *topo_size) {
    if (a->_visited) return;
    a->_visited = 1;

    if (a->_children_num)
        for (int i=0; i<a->_children_num; ++i) build_topo(a->children[i], topo, topo_size);

    Value **tmp = realloc(*topo, ((*topo_size + 1) * sizeof(**topo)));
    *topo = tmp;
    (*topo)[*topo_size] = a;
    (*topo_size)++;
}

void backward(Value *a) {
    a->grad = 1.0;

    if (!a->_children_num) return;

    // topological ordering
    Value **topo = NULL;
    size_t topo_size = 0;
    build_topo(a, &topo, &topo_size);

    for (int j=0; j < topo_size; ++j) {
        Value *v = topo[j];
        for (int i=0; i < v->_children_num; ++i) v->children[i]->grad += v->local_grads[i] * v->grad;
        // reset _visited
        v->_visited = 0;
    }
}

// --- Model Architecture ---

#define N_EMBD 16
#define N_HEAD 4
#define N_LAYER 1
#define BLOCK_SIZE 8
#define HEAD_DIM (N_EMBD / N_HEAD)

typedef struct Matrix {
    Value **data;
    
    // --- metadata ---
    int nout;
    int nin;
} Matrix;

Matrix init_matrix(int nout, int nin, float std) {
    Matrix m = {};
    m.nout = nout;
    m.nin = nin;
    m.data = malloc(nout * sizeof(Value *));
    for (int i=0; i < nout; ++i) {
        m.data[i] = malloc(nin * sizeof(Value));
        for (int j=0; j < nin; ++j) m.data[i][j] = val_from_const(random_gaussian(0.0, std));
    }
    return m;
}

void print_matrix(Matrix m) {
    printf("[\n");
    for (int i = 0; i < m.nout; ++i) {
        printf("\t");
        for (int j = 0; j < m.nin; ++j)
            printf("%f, ", m.data[i][j].data);
        printf("\n");
    }
    printf("]\n");
}

typedef struct Layer {
    Matrix attn_wq;     //
    Matrix attn_wk;     //
    Matrix attn_wv;     //
    Matrix attn_wo;     //
    Matrix mlp_fc1;     //
    Matrix mlp_fc2;     //
} Layer;

Layer init_layer() {
    Layer l = {};
    l.attn_wq = init_matrix(N_EMBD, N_EMBD, 0.02);
    l.attn_wk = init_matrix(N_EMBD, N_EMBD, 0.02);
    l.attn_wv = init_matrix(N_EMBD, N_EMBD, 0.02);
    l.attn_wv = init_matrix(N_EMBD, N_EMBD, 0.);
    l.attn_wo = init_matrix(4 * N_EMBD, N_EMBD, 0.02);
    l.mlp_fc1 = init_matrix(N_EMBD, 4 * N_EMBD, 0.0);

    return l;
}

typedef struct StateDict {
    Matrix wte;         // 
    Matrix wpe;         // 
    Matrix lm_head;     // 

    Layer *layers;      //

    // --- metadata ---
    int param_num;
} StateDict;

StateDict init_state_dict(int vocab_size) {
    StateDict sd = {};
    sd.wte = init_matrix(vocab_size, N_EMBD, 0.02);
    sd.wpe = init_matrix(BLOCK_SIZE, N_EMBD, 0.02);
    sd.lm_head = init_matrix(vocab_size, N_EMBD, 0.02);
    sd.layers = malloc(N_LAYER * sizeof(Layer));

    for (int i=0; i < N_LAYER; i++) sd.layers[i] = init_layer();

    sd.param_num =  vocab_size * N_EMBD +
                    BLOCK_SIZE * N_EMBD + 
                    vocab_size * N_EMBD +
                    N_LAYER * (
                        4 * (N_EMBD * N_EMBD) +
                        (4 * N_EMBD) * N_EMBD +
                        N_EMBD * (N_EMBD * 4)
                    );

    return sd;
}

// GPT-2 arch w/ minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU^2
Value* linear(Value *x, Matrix w) {
    // for now, assume sizes are vector/matrix sizes are ok
    Value *out = malloc(w.nout * sizeof(Value));
    for (int i=0; i < w.nout; ++i) {
        out[i] = val_from_const(0.0);
        for (int j=0; j < w.nin; ++j) {
            Value t = _mul(&w.data[i][j], &x[j]);
            out[i] = _add(&out[i], &t);
        }
    }

    return out;
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

    // 4. init params
    // int vocab_size = t.size;
    // StateDict sd = init_state_dict(vocab_size);

    int nin=2, nout=5;
    Value *x = malloc(nin * sizeof(Value));
    for (int i=0; i<nin; ++i) {
        x[i] = val_from_const((double)i);

        print_val(&x[i]);
    }

    Matrix m = init_matrix(nout, nin, 0.1);
    print_matrix(m);

    Value *l = linear(x, m);
    for (int i=0; i < nin; ++i) print_val(&l[i]);

    // debug
    // for (int i=0; i<128; i++) printf("%d ", t.items[i]);
    // printf("\n");

    free(dataset);

    printf("\n");

    return EXIT_SUCCESS;
}