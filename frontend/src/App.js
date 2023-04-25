import { useState } from 'react';
import './css/App.css';
import './css/cover.css';

function App() {
  let [barList, setBarList] = useState([true, false]);
  const [userSentence, setUserSentence] = useState("");
  const [userSubject, setUserSubject] = useState("");
  const [userObject, setUserObject] = useState("");

  const userSentenceChange = (event) => {
    setUserSentence(event.target.value);
  };
  const userSubjectChange = (event) => {
    setUserSubject(event.target.value);
  };
  const userObjectChange = (event) => {
    setUserObject(event.target.value);
  };

  return (
    <div className="App">
      <body className='text-center text-bg-dark' style={{ minHeight: "100vh" }}>
        {/* nav */}
        <div className='cover-container d-flex w-100 h-100 p-3 mx-auto flex-column'>
          <header className='mb-auto'>
            <div>
              <nav class="nav nav-masthead justify-content-center py-4">
                {barList[0] ? (
                  <a class="nav-link fw-bold py-1 px-0 active" href="#!">Intro</a>
                ) : (
                  <a class="nav-link fw-bold py-1 px-0" href="#!" onClick={() => {
                    setBarList((prev) => [true, false])
                  }}>Intro</a>
                )}
                {barList[1] ? (
                  <a class="nav-link fw-bold py-1 px-0 active" href="#!">Task</a>
                ) : (
                  <a class="nav-link fw-bold py-1 px-0" href="#!" onClick={() => {
                    setBarList((prev) => [false, true])
                  }}>Task</a>
                )}
              </nav>
            </div>
          </header>
        </div>

        <main class="container-fluid px-3">
          <div>
            {barList[0] ? (
              <div className="d-flex justify-content-left align-items-center p-5" style={{ minHeight: "85vh", marginLeft: "50px" }}>
                <div>
                  <div className="d-flex">
                    <h1 className="display-5 fw-bold py-1">Relation Extraction</h1>
                  </div>
                  <div className="d-flex">
                    <p className="lead">관계 추출은 문장의 단어에 대해 속성과 관계를 예측하여</p>
                  </div>
                  <div className="d-flex">
                    <p className="lead"><b>지식 그래프 구축</b>에 있어 핵심적인 역할을 하는 단계입니다.</p>
                  </div>
                  <div className="d-flex">
                    <p className="lead">더불어 <b>구조화된 검색, 감정 분석, 질문 답변, 요약</b>과 같은</p>
                  </div>
                  <div className="d-flex">
                    <p className="lead">다양한 자연어 처리 응용 연구 분야에서 중요하게 역할할 수 있습니다.</p>
                  </div>
                  <div className="d-flex">
                    <label
                      className="file btn btn-lg btn-light btn-primary fw-bold border-white bg-white mt-3"
                      onClick={() => {
                        setBarList((prev) => [false, true])
                      }}
                    >시작하기
                    </label>
                  </div>
                </div>
              </div>
            ) : (
              <div className='d-flex justify-content-left align-items-center p-5' style={{ minHeight: "80vh" }}>
                <div className='flex-column'>
                  <div className='text-start mb-5'>
                    <p>사용자는 문장, subject, 그리고 object를 입력할 수 있습니다.</p>
                    <p>사용자로부터 입력받은 데이터는 서버로 전송되며,</p>
                    <p>사용자는 모델이 예측한 두 속성 간 관계를 확인할 수 있습니다.</p>
                  </div>

                  <div>
                    <div className="d-flex flex-column text-start">
                      <label htmlFor="" className='mb-1'>sentence</label>
                      <input type="text" placeholder='문장을 입력하세요' className='custom-input' onChange={userSentenceChange} value={userSentence} style={{ minWidth: "50vw" }} />
                    </div>
                  </div>

                  <div className='mt-3'>
                    <div className='row'>
                      <div className="col-6">
                        <div className="d-flex flex-column text-start">
                          <label htmlFor="" className='mb-1'>subject</label>
                          <input type="text" placeholder='subject를 입력하세요' className='custom-input' onChange={userSubjectChange} style={{ width: "100%" }} />
                        </div>
                      </div>
                      <div className="col-6">
                        <div className="d-flex flex-column text-start">
                          <label htmlFor="" className='mb-1'>object</label>
                          <input type="text" placeholder='object를 입력하세요' className='custom-input' onChange={userObjectChange} style={{ width: "100%" }} />
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className='text-start mt-3'>
                    {(userSentence.length < 1 || userSubject.length < 1 || userObject.length < 1) ? (
                      <label
                        className="file btn btn-md btn-light btn-primary fw-bold border-white bg-white mt-3 disabled"
                        onClick={() => {
                          
                        }}
                      >전송하기
                      </label>
                    ): (
                      <label
                      className="file btn btn-md btn-light btn-primary fw-bold border-white bg-white mt-3"
                      onClick={() => {
                        
                      }}
                    >전송하기
                    </label>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        </main>
      </body>
    </div>
  );
}

export default App;
