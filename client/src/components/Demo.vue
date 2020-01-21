<template>
  <div class="container">
    <div class="row">
      <div class="col-sm-10">
        <h1>Keyword Extraction Demo</h1>
        <hr><br><br>
        <b-button v-on:click="onReload" type="reload" class="btn btn-success btn-sm">
          Load 10 new Sentences
        </b-button>
        <br><br>
        <button type="button" class="btn btn-success btn-sm" v-b-modal.prediction-modal>Predict own
          Sentence</button>
        <br><br>

        <table class="table table-hover">
          <thead>
            <tr>
              <th scope="col">Input</th>
              <th scope="col">Keywords</th>
              <th scope="col">Prediction</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(prediction, index) in predictions" :key="index">
              <td><span
                v-html="formatInput(prediction.input, prediction.keywords, prediction.prediction)"/>
              </td>
              <td>{{prediction.keywords}}</td>
              <td>{{prediction.prediction}}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
    <b-modal ref="addPredictionModal"
         id="prediction-modal"
         title="Add a new prediction"
         hide-footer>
      <b-form @submit="onSubmit" @reset="onReset" class="w-100">
      <b-form-group id="form-title-group"
                    label="Input:"
                    label-for="form-input-input">
          <b-form-input id="form-input-input"
                        type="text"
                        v-model="addPredictionForm.input"
                        required
                        placeholder="Enter Input">
          </b-form-input>
        </b-form-group>
        <b-button type="submit" variant="primary">Submit</b-button>
        <b-button type="reset" variant="danger">Reset</b-button>
      </b-form>
    </b-modal>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  data() {
    return {
      predictions: [],
      addPredictionForm: {
        input: '',
        keywords: '',
        prediction: '',
      },
    };
  },
  methods: {
    formatInput(inp, keywords, prediction) {
      let replacement;
      let input = inp;
      const splitKeywords = keywords.split(' ');
      for (let i = 0; i < splitKeywords.length; i += 1) {
        if (splitKeywords[i].trim().length > 1) {
          replacement = '<strong>'.concat(splitKeywords[i], '</strong>');
          // console.log(replacement);
          input = input.split(splitKeywords[i]).join(replacement);
        }
        // console.log(input);
      }
      const splitPredictions = prediction.split(' ');
      for (let i = 0; i < splitPredictions.length; i += 1) {
        if (splitPredictions[i].trim().length > 1) {
          replacement = '<u>'.concat(splitPredictions[i], '</u>');
          // console.log(replacement);
          input = input.split(splitPredictions[i]).join(replacement);
          // console.log(input);
        }
      }
      return input;
    },
    getPredictions() {
      const path = 'http://localhost:5000/predictions';
      axios.get(path)
        .then((res) => {
          this.predictions = res.data.predictions;
        })
        .catch((error) => {
          // eslint-disable-next-line
          console.error(error);
        });
    },
    addPrediction(payload) {
      const path = 'http://localhost:5000/predictions';
      axios.post(path, payload)
        .then(() => {
          this.getPredictions();
        })
        .catch((error) => {
          // eslint-disable-next-line
          console.log(error);
          this.getPredictions();
        });
    },
    initForm() {
      this.addPredictionForm.input = '';
      this.addPredictionForm.keywords = '';
      this.addPredictionForm.prediction = '';
    },
    onSubmit(evt) {
      evt.preventDefault();
      this.$refs.addPredictionModal.hide();
      const payload = {
        input: this.addPredictionForm.input,
        keywords: this.addPredictionForm.keywords,
        prediction: this.addPredictionForm.prediction,
      };
      this.addPrediction(payload);
      this.initForm();
    },
    onReset(evt) {
      evt.preventDefault();
      this.$refs.addPredictionModal.hide();
      this.initForm();
    },
    onReload(evt) {
      evt.preventDefault();
      const path = 'http://localhost:5000/predictions';
      axios.delete(path)
        .then(() => {
          this.getPredictions();
        })
        .catch((error) => {
          // eslint-disable-next-line
          console.log(error);
          this.getPredictions();
        });
    },
  },
  created() {
    this.getPredictions();
  },
};
</script>
