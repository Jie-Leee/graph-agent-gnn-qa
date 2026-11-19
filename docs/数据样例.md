# 数据样例（人工可读）

每块给出一条问答的完整字段，包括证据路径。注意 `answer_ids` 与 `evidence` 只在训练与评测时使用，推理阶段不进入提示词。

## 训练集

**Q（train-00000）** Return every time zone of any headquarters city of American International Pictures.

- 主题实体：American International Pictures (`/m/01cvxf`)
- 关系路径：`/organization/organization/headquarters./location/mailing_address/citytown >> /location/location/time_zones`
- 标准答案（1）：Pacific Time Zone (`/m/02lcqs`)
- 证据：/m/01cvxf -[/organization/organization/headquarters./location/mailing_address/citytown]-> /m/030qb3t；/m/030qb3t -[/location/location/time_zones]-> /m/02lcqs

**Q（train-00001）** List every time zone of any headquarters city of American International Pictures.

- 主题实体：American International Pictures (`/m/01cvxf`)
- 关系路径：`/organization/organization/headquarters./location/mailing_address/citytown >> /location/location/time_zones`
- 标准答案（1）：Pacific Time Zone (`/m/02lcqs`)
- 证据：/m/01cvxf -[/organization/organization/headquarters./location/mailing_address/citytown]-> /m/030qb3t；/m/030qb3t -[/location/location/time_zones]-> /m/02lcqs

**Q（train-00002）** Return every profession of any cast member of Nurse Jackie.

- 主题实体：Nurse Jackie (`/m/05p9_ql`)
- 关系路径：`/tv/tv_program/regular_cast./tv/regular_tv_appearance/actor >> /people/person/profession`
- 标准答案（3）：Professor-GB (`/m/016fly`)、Actor-GB (`/m/02hrh1q`)、Playwright-GB (`/m/02hv44_`)
- 证据：/m/05p9_ql -[/tv/tv_program/regular_cast./tv/regular_tv_appearance/actor]-> /m/07ncs0；/m/07ncs0 -[/people/person/profession]-> /m/02hrh1q；/m/05p9_ql -[/tv/tv_program/regular_cast./tv/regular_tv_appearance/actor]-> /m/0633p0；/m/0633p0 -[/people/person/profession]-> /m/02hv44_；/m/0633p0 -[/people/person/profession]-> /m/016fly

**Q（train-00003）** List every profession of any cast member of Nurse Jackie.

- 主题实体：Nurse Jackie (`/m/05p9_ql`)
- 关系路径：`/tv/tv_program/regular_cast./tv/regular_tv_appearance/actor >> /people/person/profession`
- 标准答案（3）：Professor-GB (`/m/016fly`)、Actor-GB (`/m/02hrh1q`)、Playwright-GB (`/m/02hv44_`)
- 证据：/m/05p9_ql -[/tv/tv_program/regular_cast./tv/regular_tv_appearance/actor]-> /m/07ncs0；/m/07ncs0 -[/people/person/profession]-> /m/02hrh1q；/m/05p9_ql -[/tv/tv_program/regular_cast./tv/regular_tv_appearance/actor]-> /m/0633p0；/m/0633p0 -[/people/person/profession]-> /m/02hv44_；/m/0633p0 -[/people/person/profession]-> /m/016fly

## 验证集

**Q（validation-00000）** Return every contained location of any place of birth of Robert Louis Stevenson.

- 主题实体：Robert Louis Stevenson (`/m/06lbp`)
- 关系路径：`/people/person/place_of_birth >> /location/location/contains`
- 标准答案（2）：Heriot-Watt University (`/m/030nwm`)、University of Edinburgh (`/m/0h6rm`)
- 证据：/m/06lbp -[/people/person/place_of_birth]-> /m/02m77；/m/02m77 -[/location/location/contains]-> /m/0h6rm；/m/02m77 -[/location/location/contains]-> /m/030nwm

**Q（validation-00001）** List every contained location of any place of birth of Robert Louis Stevenson.

- 主题实体：Robert Louis Stevenson (`/m/06lbp`)
- 关系路径：`/people/person/place_of_birth >> /location/location/contains`
- 标准答案（2）：Heriot-Watt University (`/m/030nwm`)、University of Edinburgh (`/m/0h6rm`)
- 证据：/m/06lbp -[/people/person/place_of_birth]-> /m/02m77；/m/02m77 -[/location/location/contains]-> /m/0h6rm；/m/02m77 -[/location/location/contains]-> /m/030nwm

**Q（validation-00002）** For Vladimir Vladimirovich Nabokov, list every contained location of any place of birth.

- 主题实体：Vladimir Vladimirovich Nabokov (`/m/03f47xl`)
- 关系路径：`/people/person/place_of_birth >> /location/location/contains`
- 标准答案（2）：Saint Petersburg Conservatory (`/m/02_gzx`)、Saint Petersburg State University (`/m/02_jjm`)
- 证据：/m/03f47xl -[/people/person/place_of_birth]-> /m/06pr6；/m/06pr6 -[/location/location/contains]-> /m/02_gzx；/m/06pr6 -[/location/location/contains]-> /m/02_jjm

**Q（validation-00003）** Return every contained location of any place of birth of Vladimir Vladimirovich Nabokov.

- 主题实体：Vladimir Vladimirovich Nabokov (`/m/03f47xl`)
- 关系路径：`/people/person/place_of_birth >> /location/location/contains`
- 标准答案（2）：Saint Petersburg Conservatory (`/m/02_gzx`)、Saint Petersburg State University (`/m/02_jjm`)
- 证据：/m/03f47xl -[/people/person/place_of_birth]-> /m/06pr6；/m/06pr6 -[/location/location/contains]-> /m/02_gzx；/m/06pr6 -[/location/location/contains]-> /m/02_jjm

## 测试集

**Q（test-00000）** For Pinocchio, list every headquarters city of any production company.

- 主题实体：Pinocchio (`/m/019kyn`)
- 关系路径：`/film/film/production_companies >> /organization/organization/headquarters./location/mailing_address/citytown`
- 标准答案（1）：Burbank (`/m/0r00l`)
- 证据：/m/019kyn -[/film/film/production_companies]-> /m/09b3v；/m/09b3v -[/organization/organization/headquarters./location/mailing_address/citytown]-> /m/0r00l

**Q（test-00001）** Return every headquarters city of any production company of Pinocchio.

- 主题实体：Pinocchio (`/m/019kyn`)
- 关系路径：`/film/film/production_companies >> /organization/organization/headquarters./location/mailing_address/citytown`
- 标准答案（1）：Burbank (`/m/0r00l`)
- 证据：/m/019kyn -[/film/film/production_companies]-> /m/09b3v；/m/09b3v -[/organization/organization/headquarters./location/mailing_address/citytown]-> /m/0r00l

**Q（test-00002）** List every religion of any writer of High Fidelity.

- 主题实体：High Fidelity (`/m/025rxjq`)
- 关系路径：`/film/film/written_by >> /people/person/religion`
- 标准答案（1）：Catholicism (`/m/0c8wxp`)
- 证据：/m/025rxjq -[/film/film/written_by]-> /m/01gkmx；/m/01gkmx -[/people/person/religion]-> /m/0c8wxp

**Q（test-00003）** Name every religion of any writer of High Fidelity.

- 主题实体：High Fidelity (`/m/025rxjq`)
- 关系路径：`/film/film/written_by >> /people/person/religion`
- 标准答案（1）：Catholicism (`/m/0c8wxp`)
- 证据：/m/025rxjq -[/film/film/written_by]-> /m/01gkmx；/m/01gkmx -[/people/person/religion]-> /m/0c8wxp
